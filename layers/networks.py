import numpy as np

import tensorflow as tf
from tensorflow.python.keras import layers

import tensorflow_probability as tfp

from layers import bayes_layers, layers
import mvg_dist.distributions as mvg_dist

tfd = tfp.distributions
ed = tfp.edward2


def make_encoder(y, latent_d, hidden_size=(500, 500)):
    net = make_dense(y, latent_d * 2, {"hidden_size": hidden_size}, concrete=False)
    dist = mvg_dist.MultivariateNormalLogDiag(
        loc=net[..., :latent_d],
        log_covariance_diag=net[..., latent_d:],
        name="encoder"
    )

    return dist


def make_decoder(x_code,
                 y_shape,
                 architecture_params,
                 architecture="bayes_dense",
                 activation="softplus",
                 output_distribution="bernoulli"):

    possible_architectures = ["bayes_dense", "bayes_conv", "regular", "concrete"]
    possible_output_dists = ["bernoulli", "gaussian"]
    possible_activations = ["softplus", "sigmoid", "relu", "tanh"]

    assert architecture in possible_architectures
    assert output_distribution in possible_output_dists
    assert activation in possible_activations

    if output_distribution == "gaussian":
        output_shape = np.prod(y_shape) * 2
    else:
        output_shape = np.prod(y_shape)

    if activation == "softplus":
        activ_function = tf.nn.softplus
    elif activation == "relu":
        activ_function = tf.nn.relu
    elif activation == "tanh":
        activ_function = tf.nn.tanh
    else:
        activ_function = tf.nn.sigmoid

    if architecture == "bayes_dense":
        net = make_bayes_dense(x_code, output_shape, architecture_params, activ_function)
    elif architecture == "bayes_conv":
        net = make_bayes_conv(x_code, y_shape, architecture_params, activ_function)
    elif architecture == "concrete":
        net = make_dense(x_code, output_shape, architecture_params, activ_function, concrete=True)
    else:
        net = make_dense(x_code, output_shape, architecture_params, activ_function, concrete=False)

    dimensions = len(y_shape)
    if output_distribution == "gaussian":
        means = tf.reshape(
            net[..., :np.prod(y_shape)], tf.concat([[-1], y_shape], axis=0),
            name="means"
        )
        log_variance = tf.reshape(net[..., np.prod(y_shape):], tf.concat([[-1], y_shape], axis=0), name="log_variance")
        return mvg_dist.MultivariateNormalLogDiag(loc=means,
                                                  log_covariance_diag=log_variance)
    else:
        logits = tf.reshape(
            net, tf.concat([[-1], y_shape], axis=0)
        )
        return tfd.Independent(tfd.Bernoulli(logits=logits, dtype=tf.float64), dimensions)


def make_dense(x, out_size, architecture_params, activation=tf.nn.sigmoid, concrete=True):
    model_layers = [tf.keras.layers.Flatten()]
    for h in architecture_params["hidden_size"]:
        if concrete:
            model_layers.append(layers.ConcreteDropout(tf.keras.layers.Dense(h, activation=activation), trainable=True))
        else:
            model_layers.append(tf.keras.layers.Dense(h, activation=activation))
    if concrete:
        model_layers.append(layers.ConcreteDropout(tf.keras.layers.Dense(out_size), trainable=True))
    else:
        model_layers.append(tf.keras.layers.Dense(out_size))
    model = tf.keras.Sequential(model_layers)
    net = model(x, training=True)
    return net


def make_bayes_dense(x, out_size, architecture_params, activation=tf.nn.softplus):
    model_layers = [tf.keras.layers.Flatten()]
    for h in architecture_params["hidden_size"]:
        model_layers.append(tfp.layers.DenseReparameterization(
            h,
            activation=activation,
            kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
            kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))
    model_layers.append(tfp.layers.DenseReparameterization(
        np.array(out_size, dtype=np.int32),
        kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
        kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))
    model = tf.keras.Sequential(model_layers)
    net = model(x)
    regularizer = tf.reduce_sum(model.losses)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return net


def make_bayes_conv(x, out_size, architecture_params, activation=tf.nn.softplus):
    kernel_size = architecture_params["kernel_size"]
    strides = architecture_params["strides"]
    padding = architecture_params["padding"]
    filters = architecture_params["filters"]

    input_shape = x.shape.as_list()
    if len(input_shape) == 2:
        net = tf.matrix_diag(x)
        net = tf.expand_dims(net, axis=-1)
    elif len(input_shape) == 3:
        net = tf.expand_dims(x, axis=-1)
    elif len(input_shape) == 4:
        net = x
    else:
        raise RuntimeError(f'Input as invalid number of dimensions ({len(input_shape)}).' +
                           ' Valid numbers are 2, 3 and 4')
    model_layers = []

    for i, (s, f) in enumerate(zip(strides, filters)):
        if i == len(filters) - 1:
            break
        model_layers.append(bayes_layers.Conv2DTransposeReparameterization(
            f, kernel_size, s, padding, activation=activation,
            kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
            kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))

    model_layers.append(bayes_layers.Conv2DTransposeReparameterization(
        filters[-1], kernel_size, strides[-1], padding,
        kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
        kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))

    model = tf.keras.Sequential(model_layers)

    net = model(net)
    output_shape = net.shape.as_list()
    if output_shape[1] < out_size[0]:
        h_pad_up = (out_size[0] - output_shape[1]) // 2
        h_pad_down = out_size[0] - output_shape[1] - h_pad_up
        net = tf.pad(net, [[0, 0], [h_pad_up, h_pad_down], [0, 0], [0, 0]])
    else:
        net = net[:, :out_size[0], :, :]

    if output_shape[2] < out_size[1]:
        w_pad_up = (out_size[0] - output_shape[2]) // 2
        w_pad_down = out_size[0] - output_shape[2] - w_pad_up
        net = tf.pad(net, [[0, 0], [0, 0], [w_pad_up, w_pad_down], [0, 0]])
    else:
        net = net[:, :, :out_size[1], :]

    regularizer = tf.reduce_sum(model.losses)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return net
