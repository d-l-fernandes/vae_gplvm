import tensorflow as tf
import tensorflow_probability as tfp

import base_model as bm
import mvg_dist.distributions as mvg_dist
from gp import kernels, gps
from layers import networks

ed = tfp.edward2
tfd = tfp.distributions


class VAEGPLVMModel(bm.BaseModel):
    def __init__(self, config):
        super(VAEGPLVMModel, self).__init__(config)

        self.t_x = None
        self.t_y = None
        self.t_z = None

        self.gp = None
        self.t_encoder = None
        self.t_decoder = None
        self.p_x = None

        self.t_square_loss = None

        self.t_full_reco = None
        self.t_avg_reco = None
        self.t_avg_kl_local = None
        self.t_avg_kl_global = None
        self.t_avg_elbo_loss = None
        self.hyperprior = None

        self.opt_trainer_pretraining = None
        self.opt_trainer_kernels = None
        self.opt_trainer_global = None
        self.opt_trainer_mlp = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        input_shape = [self.config["batch_size"]] + self.config["state_size"]
        input_shape_pca = [self.config["batch_size"]] + [self.config["gp_q"]]

        # Data
        self.t_y = tf.placeholder(shape=input_shape, dtype=self.dtype, name='Y')
        self.t_x = tf.placeholder(shape=input_shape_pca, dtype=self.dtype, name='X')

        # Dimensions
        t_n_batch = self.config["batch_size"]
        t_gp_q = self.config["gp_q"]
        t_vae_q = self.config["vae_q"]
        m_beta = self.config["num_ind_points_beta"]
        m_gamma = self.config["num_ind_points_gamma"]

        weight = t_n_batch / self.config["num_data_points"]

        with tf.name_scope("kernels"):
            log_amplitude_gp = tf.Variable(-2. * tf.ones([t_vae_q], dtype=self.dtype), name="log_amplitude_latent")
            log_kernel_weights_gp = tf.Variable(tf.zeros([t_vae_q, t_gp_q], dtype=self.dtype),
                                                name="log_kernel_weights_latent")
            latent_kernel = kernels.ExponentiatedQuadratic(log_amplitude_gp, log_kernel_weights_gp)
            self.hyperprior = \
                0.5 * (tf.reduce_sum(tf.square(log_amplitude_gp)) +
                       tf.reduce_sum(tf.square(log_kernel_weights_gp)))

        with tf.name_scope("gp"):
            self.gp = gps.OrthogonallyDecoupledGP(latent_kernel, t_gp_q, t_vae_q, m_beta, m_gamma)

        with tf.name_scope("encoder"):
            # Encoder
            self.t_encoder = networks.make_encoder(self.t_y, self.config["gp_q"],
                                                   self.config["encoder_hidden_size"])
            self.t_square_loss = \
                tf.square(tf.reduce_mean(self.t_encoder.mean() - self.t_x)) + \
                tf.square(tf.reduce_mean(
                    self.t_encoder.covariance() - tf.tile([tf.zeros([t_gp_q, t_gp_q], dtype=self.dtype)],
                                                          [t_n_batch, 1, 1])))

            self.p_x = mvg_dist.MultivariateNormalLogDiag(loc=tf.zeros([t_n_batch, t_gp_q]), name="p_X")

        self.t_z = tf.transpose(self.gp.predict(self.t_encoder.sample(), diag_covariance=True).sample())

        with tf.name_scope("decoder"):
            self.t_decoder = networks.make_decoder(
                self.t_z,
                y_shape=tuple(input_shape[1:]),
                architecture_params=self.config["architecture_params"],
                architecture=self.config["architecture"],
                activation=self.config["activation"],
                output_distribution=self.config["output_distribution"]
            )

        with tf.name_scope("elbo"):
            self.t_full_reco = self.t_decoder.log_prob(self.t_y)
            self.t_avg_reco = tf.reduce_mean(self.t_full_reco)
            self.t_avg_kl_global = weight * self.gp.regularization
            self.t_avg_kl_local = tf.reduce_mean(self.t_encoder.kl_divergence(self.p_x))
            self.t_avg_elbo_loss = \
                self.t_avg_reco - self.t_avg_kl_local - self.t_avg_kl_global - self.hyperprior - \
                tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) / \
                tf.constant(self.config["num_data_points"], dtype=self.dtype)

        with tf.name_scope("trainers"):
            self.opt_trainer_pretraining = tf.contrib.opt.NadamOptimizer(
                learning_rate=self.config["learning_rate_mlp"]).minimize(
                self.t_square_loss, var_list=tf.trainable_variables("encoder"))
            self.opt_trainer_kernels = tf.contrib.opt.NadamOptimizer(
                learning_rate=self.config["learning_rate_kernels"]).minimize(
                -self.t_avg_elbo_loss, var_list=tf.trainable_variables("kernels"))
            self.opt_trainer_global = tf.contrib.opt.NadamOptimizer(
                learning_rate=self.config["learning_rate_global"]).minimize(
                -self.t_avg_elbo_loss, var_list=tf.trainable_variables("gp"))
            self.opt_trainer_mlp = tf.contrib.opt.NadamOptimizer(
                learning_rate=self.config["learning_rate_mlp"]).minimize(
                -self.t_avg_elbo_loss,
                var_list=tf.trainable_variables("encoder") + tf.trainable_variables("decoder"))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
