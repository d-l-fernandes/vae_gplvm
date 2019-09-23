import numpy as np
import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util


def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
    return result


class ExponentiatedQuadratic(psd_kernel.PositiveSemidefiniteKernel):

    def __init__(self,
                 amplitude=None,
                 length_scale=None,
                 use_log_squared=True,
                 feature_ndims=1,
                 validate_args=False,
                 name='ExponentiatedQuadratic'):
        """Construct an ExponentiatedQuadratic kernel instance.
        Args:
          amplitude: floating point `Tensor` that controls the maximum value
            of the kernel. Must have shape () (single output) or (d_out) (multi output)
          length_scale: floating point `Tensor` that controls how sharp or wide the
            kernel shape is. Must have shape () (no ard) or (d_in) (with ard)
          use_log_squared: If `True`, uses takes exponent of amplitude and length_scale
            divided by 2. Also, does not check if positive
          feature_ndims: Python `int` number of rightmost dims to include in the
            squared difference norm in the exponential.
          validate_args: If `True`, parameters are checked for validity despite
            possibly degrading runtime performance
          name: Python `str` name prefixed to Ops created by this class.
        """
        with tf.name_scope(name, values=[amplitude, length_scale]) as name:
            dtype = dtype_util.common_dtype([amplitude, length_scale], tf.float32)
            if amplitude is not None:
                amplitude = tf.convert_to_tensor(
                    amplitude, name='amplitude', dtype=dtype)
            else:
                if use_log_squared:
                    amplitude = tf.constant(0., name='amplitude', dtype=dtype)
                else:
                    amplitude = tf.constant(1., name='amplitude', dtype=dtype)
            if not use_log_squared:
                self._amplitude = _validate_arg_if_not_none(
                    amplitude, tf.assert_positive, validate_args)
            else:
                self._amplitude = amplitude

            if length_scale is not None:
                length_scale = tf.convert_to_tensor(
                    length_scale, name='length_scale', dtype=dtype)
            else:
                if use_log_squared:
                    length_scale = tf.constant(0., name='amplitude', dtype=dtype)
                else:
                    length_scale = tf.constant(1., name='amplitude', dtype=dtype)
            if not use_log_squared:
                self._length_scale = _validate_arg_if_not_none(
                    length_scale, tf.assert_positive, validate_args)
            else:
                self._length_scale = length_scale
            tf.debugging.assert_same_float_dtype([self._amplitude, self._length_scale])

        self.use_log_squared = use_log_squared
        super(ExponentiatedQuadratic, self).__init__(
            feature_ndims, dtype=dtype, name=name)

    @property
    def amplitude(self):
        """Amplitude parameter."""
        if self.use_log_squared:
            return tf.exp(self._amplitude / 2.)
        else:
            return self._amplitude

    @property
    def length_scale(self):
        """Length scale parameter."""
        if self.use_log_squared:
            return tf.exp(self._length_scale / 2.)
        else:
            return self._length_scale

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape if self.amplitude is None else self.amplitude.shape

    def _batch_shape_tensor(self):
        return [] if self.amplitude is None else tf.shape(self.amplitude)

    def _apply(self, x1, x2, param_expansion_ndims=1):
        length_scale = tf.expand_dims(tf.expand_dims(self.length_scale, -2), -2)
        exponent = tf.exp(-0.5 * tf.reduce_sum(length_scale**2 * tf.math.squared_difference(x1, x2), axis=-1))

        amplitude = util.pad_shape_with_ones(self.amplitude, param_expansion_ndims)
        exponent *= amplitude**2

        return exponent
