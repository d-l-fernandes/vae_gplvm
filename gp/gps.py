import sys
sys.path.insert(0, '..')
import tensorflow as tf
import tensorflow_probability as tfp
import mvg_dist.distributions as mvg_dist

tfd = tfp.distributions


class OrthogonallyDecoupledGP:

    def __init__(self,
                 kernel,
                 ndims_in,
                 ndims_out,
                 n_inducing_beta=300,
                 n_inducing_gamma=800,
                 input_variables=None,
                 time=None,
                 name="OrthogonallyDecoupledGP"):

        if not isinstance(input_variables, dict):
            input_variables = dict()
        else:
            possible_keys = ["log_noise_variance", "x_u_beta", "x_u_gamma", "mu_u_beta", "l_u_beta", "mu_u_gamma"]
            wrong_keys = []
            for k in input_variables.keys():
                if k not in possible_keys:
                    wrong_keys.append(k)

            if wrong_keys:
                raise ValueError("Invalid key(s) in input_variables dict", wrong_keys)

        with tf.name_scope(name, values=[ndims_in, ndims_out, n_inducing_beta, n_inducing_gamma]):
            self.dtype = tf.float32

            # Dimensions
            self.ndims_in = tf.convert_to_tensor(ndims_in, dtype=tf.int32, name="ndims_in")
            self.ndims_out = tf.convert_to_tensor(ndims_out, dtype=tf.int32, name="ndims_in")
            self.n_inducing_beta = tf.convert_to_tensor(n_inducing_beta, dtype=tf.int32, name="n_inducing_beta")
            self.n_inducing_gamma = tf.convert_to_tensor(n_inducing_gamma, dtype=tf.int32, name="n_inducing_gamma")
            self.n_lower_inducing_beta = tf.convert_to_tensor(
                (1 + n_inducing_beta) * n_inducing_beta // 2, dtype=tf.int32, name="n_lower_inducing_beta")

            jitter = 1.e-5

            if time is not None:
                time = tf.constant(time, self.dtype)

            # Kernel
            self.kernel = kernel

            if "log_noise_variance" in input_variables:
                self.log_noise_variance = input_variables["log_noise_variance"]
            else:
                # self.log_noise_variance = tf.Variable(tf.zeros([self.ndims_out], dtype=self.dtype),
                #                                       name="log_noise_variance")
                self.log_noise_variance = tf.Variable(-3., dtype=self.dtype, name="log_noise_variance")

            # Global variational parameters
            # Beta
            if "mu_u_beta" in input_variables:
                mu_u_beta = input_variables["mu_u_beta"]
            else:
                mu_u_beta = tf.Variable(tf.random.normal([self.ndims_out, self.n_inducing_beta], dtype=self.dtype),
                                        name="mu_u_beta")
            if "l_u_beta" in input_variables:
                self.l_u_beta = input_variables["l_u_beta"]
            else:
                random_shape = [self.ndims_out, self.n_lower_inducing_beta]
                s_u_beta = tf.Variable(tf.random.normal(random_shape,dtype=self.dtype), name="S_u_beta")
                self.l_u_beta = tfp.distributions.fill_triangular(s_u_beta)

            # if len(mu_u_beta.shape) == 3 and len(self.l_u_beta.shape) == 3:
            #     self.l_u_beta = tf.tile([self.l_u_beta], [mu_u_beta.shape[0], 1, 1, 1])
            # Gamma
            if "mu_u_gamma" in input_variables:
                mu_u_gamma = input_variables["mu_u_gamma"]
            else:
                mu_u_gamma = tf.Variable(tf.random.normal([self.ndims_out, self.n_inducing_gamma], dtype=self.dtype),
                                         name="mu_u_gamma")

            # Inducing points
            if "x_u_beta" in input_variables:
                self.x_u_beta = input_variables["x_u_beta"]
            else:
                if time is None:
                    self.x_u_beta = tf.Variable(tf.random.normal([self.n_inducing_beta, self.ndims_in],
                                                                 dtype=self.dtype),
                                                name="X_u_beta")
                else:
                    x_beta = tf.Variable(tf.random.normal([self.n_inducing_beta, self.ndims_in], dtype=self.dtype))
                    t_beta = tf.math.log(time) * tf.Variable(tf.random.uniform([self.n_inducing_beta, 1]))
                    self.x_u_beta = tf.concat([x_beta, tf.exp(t_beta)], axis=1, name="X_u_beta")
            if "x_u_gamma" in input_variables:
                self.x_u_gamma = input_variables["x_u_gamma"]
            else:
                if time is None:
                    self.x_u_gamma = tf.Variable(tf.random.normal([self.n_inducing_gamma, self.ndims_in],
                                                                  dtype=self.dtype),
                                                 name="X_u_gamma")
                else:
                    x_gamma = tf.Variable(tf.random.normal([self.n_inducing_gamma, self.ndims_in], dtype=self.dtype))
                    t_gamma = tf.math.log(time) * tf.Variable(tf.random.uniform([self.n_inducing_gamma, 1]))
                    self.x_u_gamma = tf.concat([x_gamma, tf.exp(t_gamma)], axis=1, name="X_u_gamma")

            with tf.name_scope("kernel"):
                self.k_beta_beta = self.kernel.matrix(self.x_u_beta, self.x_u_beta)
                self.l_beta_beta = tf.linalg.cholesky(self.k_beta_beta +
                                                      jitter * tf.eye(self.n_inducing_beta, dtype=self.dtype))

                k_gamma_gamma = \
                    self.kernel.matrix(self.x_u_gamma, self.x_u_gamma) + \
                    jitter * tf.eye(self.n_inducing_gamma, dtype=self.dtype)

                k_beta_gamma = self.kernel.matrix(self.x_u_beta, self.x_u_gamma)

                inv_k_beta_k_beta_gamma = tf.linalg.cholesky_solve(self.l_beta_beta, k_beta_gamma)

            with tf.name_scope("diagonal_gamma_kernel"):
                k_gb_k_b_inv_k_bg_diag = \
                    tf.reduce_sum(k_beta_gamma * inv_k_beta_k_beta_gamma, axis=1)
                d_gamma_given_beta = tf.linalg.diag_part(
                    k_gamma_gamma - tf.expand_dims(k_gb_k_b_inv_k_bg_diag, axis=-1), name="t_D_gamma")

            with tf.name_scope("a_z"):
                self.a_gamma = mu_u_gamma / d_gamma_given_beta

                if len(mu_u_beta.shape) == 3:
                    n_tile = mu_u_beta.shape[0]
                    self.a_beta = \
                        tf.reduce_sum(tf.linalg.cholesky_solve(tf.tile([self.l_beta_beta], [n_tile, 1, 1, 1]),
                                                               tf.linalg.diag(mu_u_beta)), axis=-1) - \
                        tf.einsum('cab,dcb->dca', inv_k_beta_k_beta_gamma, self.a_gamma)
                else:
                    self.a_beta = \
                        tf.reduce_sum(tf.linalg.cholesky_solve(self.l_beta_beta,
                                                               tf.linalg.diag(mu_u_beta)), axis=-1) - \
                        tf.einsum('cab,cb->ca', inv_k_beta_k_beta_gamma, self.a_gamma)

            with tf.name_scope("U_beta_dists"):
                self.q_u_beta = mvg_dist.MultivariateNormalTriLLogDiagonal(mu_u_beta,
                                                                           self.l_u_beta, name="q_u_beta")
                self.p_u_beta = mvg_dist.MultivariateNormalTriLLogDiagonal(
                    tf.zeros([self.ndims_out, self.n_inducing_beta]),
                    tfd.matrix_diag_transform(self.l_beta_beta, tf.math.log),
                    name="p_u_beta")

            with tf.name_scope("U_gamma_dists"):
                if len(mu_u_beta.shape) == 3:
                    inv_k_beta_beta_u_beta = tf.reduce_sum(
                        tf.linalg.cholesky_solve(tf.tile([self.l_beta_beta], [n_tile, 1, 1, 1]),
                                                 tf.linalg.diag(self.q_u_beta.sample())),
                        axis=-1
                    )
                    u_gamma_given_beta_mean = tf.einsum('kij,aki->akj', k_beta_gamma, inv_k_beta_beta_u_beta)
                else:
                    inv_k_beta_beta_u_beta = tf.reduce_sum(
                        tf.linalg.cholesky_solve(self.l_beta_beta, tf.linalg.diag(self.q_u_beta.sample())),
                        axis=-1
                    )
                    u_gamma_given_beta_mean = tf.einsum('kij,ki->kj', k_beta_gamma, inv_k_beta_beta_u_beta)
                self.p_u_gamma_given_beta = mvg_dist.MultivariateNormalLogDiag(
                    u_gamma_given_beta_mean,
                    tf.math.log(d_gamma_given_beta),
                    name="p_u_gamma_given_beta"
                )
                self.q_u_gamma_given_beta = mvg_dist.MultivariateNormalLogDiag(
                    u_gamma_given_beta_mean+mu_u_gamma,
                    tf.math.log(d_gamma_given_beta),
                    name="q_u_gamma_given_beta"
                )

    @property
    def regularization(self):
        reg = \
            tf.reduce_mean(self.q_u_beta.kl_divergence(self.p_u_beta)) \
            + tf.reduce_mean(self.q_u_gamma_given_beta.kl_divergence(self.p_u_gamma_given_beta))
        return reg

    def predict(self, x_input, diag_covariance=False):

        with tf.name_scope("predict_kernel"):
            k_beta_x = self.kernel.matrix(self.x_u_beta, x_input)
            k_gamma_x = self.kernel.matrix(self.x_u_gamma, x_input)
            k_x_x = self.kernel.matrix(x_input, x_input)

        with tf.name_scope("posterior_mean"):
            if len(self.a_gamma.shape) == 3:
                mu_gamma_beta = \
                    tf.einsum('cab,bca->bc', k_gamma_x, self.a_gamma) + \
                    tf.einsum('cab,bca->bc', k_beta_x, self.a_beta)
            else:
                mu_gamma_beta = \
                    tf.einsum('cab,ca->bc', k_gamma_x, self.a_gamma) + \
                    tf.einsum('cab,ca->bc', k_beta_x, self.a_beta)

        with tf.name_scope("posterior_covariance"):
            aux_matrix1 = tf.linalg.triangular_solve(self.l_beta_beta, k_beta_x, lower=True)
            aux_matrix2 = tf.linalg.triangular_solve(tf.transpose(self.l_beta_beta, [0, 2, 1]),
                                                     aux_matrix1, lower=False)
            s_minus_k_beta = tf.matmul(tfd.matrix_diag_transform(self.l_u_beta, tf.exp),
                                       tfd.matrix_diag_transform(self.l_u_beta, tf.exp),
                                       transpose_b=True) - self.k_beta_beta
            if len(self.a_gamma.shape) == 3:
                aux_matrix3 = tf.einsum('abcd,bda->bca', s_minus_k_beta, aux_matrix2)
            else:
                aux_matrix3 = tf.matmul(s_minus_k_beta, aux_matrix2)

            if diag_covariance:
                delta_cov = tf.reduce_sum(aux_matrix2 * aux_matrix3, 1) # For diagonal
                s = tf.matrix_diag_part(k_x_x) + delta_cov + tf.exp(self.log_noise_variance)
                likelihood = mvg_dist.MultivariateNormalLogDiag(
                    tf.transpose(mu_gamma_beta),
                    tf.math.log(s),
                    name="likelihood"
                )
            else:
                delta_cov = tf.matmul(aux_matrix2, aux_matrix3, transpose_a=True)
                s = k_x_x + delta_cov + tf.exp(self.log_noise_variance) * tf.eye(tf.shape(k_x_x)[-1], dtype=self.dtype)
                likelihood = mvg_dist.MultivariateNormalTriLLogDiagonal(
                    tf.transpose(mu_gamma_beta),
                    tfd.matrix_diag_transform(tf.linalg.cholesky(s), tf.math.log),
                    name="likelihood"
                )

        return likelihood
