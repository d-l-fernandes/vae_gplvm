import numpy as np
from tqdm import tqdm

import base_model as bm


class VAEGPLVMTrainer(bm.BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(VAEGPLVMTrainer, self).__init__(sess, model, data, config, logger)
        input_shape = [self.config["batch_size"]] + self.config["state_size"]
        input_pca_shape = [self.config["batch_size"]] + [self.config["gp_q"]]
        self.sess.run(self.init,
                      feed_dict={self.model.t_y: np.ones(input_shape),
                                 self.model.t_x: np.ones(input_pca_shape)})
        self.batch_gen = None

    def train_epoch(self, cur_epoch):
        # name of metric: [variable, if mean is to be used]
        metrics_pretraining = {
            "square_loss": [[], True],
        }
        metrics = {
            "elbo": [[], True],
            "reco": [[], True],
            "hist_reco": [[], False],
            "kl_local": [[], True],
            "kl_global": [[], True],
        }

        if cur_epoch == 0:
            print("Training MPL")
            self.batch_gen = self.data.select_batch_generator("training")
            for e in range(self.config["epochs_mlp"]):
                loop = tqdm(range(self.config["num_iter_per_epoch"]), desc=f"Epoch {e+1}/{self.config['epochs_mlp']}",
                            ascii=True)
                for _ in loop:
                    metrics_step = self.train_step_mlp()
                    metrics_pretraining = self.update_metrics_dict(metrics_pretraining, metrics_step)

                summaries_dict = self.create_summaries_dict(metrics_pretraining)
                self.logger.summarize(e+1, summaries_dict=summaries_dict)
            print("Finished training MPL")

        self.batch_gen = self.data.select_batch_generator("training")
        loop = tqdm(range(self.config["num_iter_per_epoch"]), desc=f"Epoch {cur_epoch+1}/{self.config['num_epochs']}",
                    ascii=True)

        for _ in loop:
            metrics_step = self.train_step()
            metrics = self.update_metrics_dict(metrics, metrics_step)

        summaries_dict = self.create_summaries_dict(metrics)

        self.logger.summarize(cur_epoch+1, summaries_dict=summaries_dict)
        self.model.save(self.sess, summaries_dict["Metrics/elbo"])

        return summaries_dict["Metrics/elbo"]

    def train_step_mlp(self):
        batch_y, batch_x = next(self.batch_gen)
        feed_dict = {self.model.t_y: batch_y, self.model.t_x: batch_x}

        cost = None
        for j in range(self.config["pretraining_iterations"]):
            _, cost = self.sess.run((self.model.opt_trainer_pretraining, self.model.t_square_loss), feed_dict)

        metrics = {
            "square_loss": cost,
        }
        return metrics

    def train_step(self):
        batch_y, batch_x = next(self.batch_gen)
        feed_dict = {self.model.t_y: batch_y}

        # Train Encoder
        _ = self.sess.run(self.model.opt_trainer_encoder, feed_dict)

        for i in range(self.config["global_iterations"]):
            _, _ = \
                self.sess.run((
                    self.model.opt_trainer_global,
                    self.model.opt_trainer_kernels),
                    feed_dict)

        _,  cost, reco, reco_full, kl_local, kl_global = \
            self.sess.run((
                self.model.opt_trainer_decoder,
                self.model.t_avg_elbo_loss,
                self.model.t_avg_reco,
                self.model.t_full_reco,
                self.model.t_avg_kl_local,
                self.model.t_avg_kl_global),
                feed_dict)

        metrics = {
            "elbo": cost,
            "reco": reco,
            "hist_reco": reco_full,
            "kl_local": kl_local,
            "kl_global": kl_global,
        }

        return metrics


class VAEGPLVMRegressionTrainer(bm.BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(VAEGPLVMRegressionTrainer, self).__init__(sess, model, data, config, logger)
        input_shape = [self.config["batch_size"]] + self.config["state_size"]
        input_pca_shape = [self.config["batch_size"]] + [self.config["gp_q"]]
        self.sess.run(self.init,
                      feed_dict={self.model.t_y: np.ones(input_shape),
                                 self.model.t_x: np.ones(input_pca_shape)})
        self.batch_gen = None

    def train_epoch(self, cur_epoch):
        # name of metric: [variable, if mean is to be used]
        metrics = {
            "elbo": [[], True],
            "reco": [[], True],
            "kl_global": [[], True],
        }

        self.batch_gen = self.data.select_batch_generator("training")
        loop = tqdm(range(self.config["num_iter_per_epoch"]), desc=f"Epoch {cur_epoch+1}/{self.config['num_epochs']}",
                    ascii=True)

        for _ in loop:
            metrics_step = self.train_step()
            metrics = self.update_metrics_dict(metrics, metrics_step)

        summaries_dict = self.create_summaries_dict(metrics)

        self.logger.summarize(cur_epoch+1, summaries_dict=summaries_dict)
        self.model.save(self.sess, summaries_dict["Metrics/elbo"])

        return summaries_dict["Metrics/elbo"]

    def train_step(self):
        batch_y, batch_x = next(self.batch_gen)
        feed_dict = {self.model.t_y: batch_y, self.model.t_x: batch_x}

        cost = None
        reco = None
        kl_global = None
        for i in range(self.config["global_iterations"]):
            _, _, _, cost, reco, kl_global = \
                self.sess.run((
                    self.model.opt_trainer_global,
                    self.model.opt_trainer_kernels,
                    self.model.opt_trainer_decoder,
                    self.model.t_avg_elbo_loss,
                    self.model.t_avg_reco,
                    self.model.t_avg_kl_global),
                    feed_dict)
        metrics = {
            "elbo": cost,
            "reco": reco,
            "kl_global": kl_global,
        }

        return metrics
