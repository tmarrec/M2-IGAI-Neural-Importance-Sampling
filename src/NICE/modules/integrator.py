""" Implement the flow integrator. """

import tensorflow as tf
import tensorflow_probability as tfp

from modules import divergences

# pylint: disable=invalid-name
tfb = tfp.bijectors
tfd = tfp.distributions


# pylint: enable=invalid-name


class Integrator():
    """ Class implementing a normalizing flow integrator.

    Args:
        - func: Function to be integrated
        - dist: Distribution to be trained to match the function
        - optimizer: An optimizer from tensorflow used to train the network
        - loss_func: The loss function to be minimized
        - kwargs: Additional arguments that need to be passed to the loss

    """

    def __init__(self, dist, optimizer, loss_func='chi2', **kwargs):
        """ Initialize the normalizing flow integrator. """
        self.global_step = 0
        self.dist = dist
        self.optimizer = optimizer
        self.divergence = divergences.Divergence(**kwargs)
        self.loss_func = self.divergence(loss_func)
        self.ckpt_manager = None

    def manager(self, ckpt_manager):
        """ Set the check point manager """
        self.ckpt_manager = ckpt_manager

    @tf.function
    def train_one_step(self, paths, probas, integral=False):
        """ Perform one step of integration and improve the sampling.

        Args:
            - paths: samples to be used for a training step
            - probas: probabilities associated with the samples
            - integral(bool): Flag for returning the integral value or not.

        Returns:
            - loss: Value of the loss function for this step
            - integral (optional): Estimate of the integral value
            - uncertainty (optional): Integral statistical uncertainty

        """

        with tf.GradientTape() as tape:
            test = self.dist.prob(paths)
            logq = self.dist.log_prob(paths)
            mean, var = tf.nn.moments(x=probas / test, axes=[0])
            true = tf.stop_gradient(probas / mean)
            logp = tf.where(true > 1e-16, tf.math.log(true),
                            tf.math.log(true + 1e-16))
            loss = self.loss_func(true, test, logp, logq)

        grads = tape.gradient(loss, self.dist.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.dist.trainable_variables))

        if integral:
            return loss, mean, tf.sqrt(var / (len(paths) - 1.))

        return loss

    @tf.function
    def sample(self, nsamples):
        """ Sample from the trained distribution.

        Args:
            nsamples(int): Number of points to be sampled.

        Returns:
            tf.tensor of size (nsamples, ndim) of sampled points.

        """
        return self.dist.sample(nsamples)

    @tf.function
    def sample_weights(self, paths):

        return self.dist.prob(paths)

    def save_weights(self):
        """ Save the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.save_weights(
                './models/model_layer_{:02d}'.format(j))

    def load_weights(self):
        """ Load the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.load_weights(
                './models/model_layer_{:02d}'.format(j))
        print("Model loaded successfully")

    def save(self):
        """ Function to save a checkpoint of the model and optimizer,
            as well as any other trackables in the checkpoint.
            Note that the network architecture is not saved, so the same
            network architecture must be used for the same saved and loaded
            checkpoints (network arch can be saved if required).
        """

        if self.ckpt_manager is not None:
            save_path = self.ckpt_manager.save()
            print("Saved checkpoint at: {}".format(save_path))
        else:
            print("There is no checkpoint manager supplied for saving the "
                  "network weights, optimizer, or other trackables.")
            print("Therefore these will not be saved and the training will "
                  "start from default values in the future.")
            print("Consider using a checkpoint manager to save the network "
                  "weights and optimizer.")

    @staticmethod
    def load(loadname, checkpoint=None):
        """ Function to load a checkpoint of the model, optimizer,
            and any other trackables in the checkpoint.

            Note that the network architecture is not saved, so the same
            network architecture must be used for the same saved and loaded
            checkpoints. Network arch can be loaded if it is saved.

        Args:
            loadname (str) : The postfix of the directory where the checkpoints
                             are saved, e.g.,
                             ckpt_dir = "./models/tf_ckpt_" + loadname + "/"
            checkpoint (object): tf.train.checkpoint instance.
        Returns:
            Nothing returned.

        """
        ckpt_dir = "./models/tf_ckpt_" + loadname + "/"
        if checkpoint is not None:
            status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
            status.assert_consumed()
            print("Loaded checkpoint")
        else:
            print("Not Loading any checkpoint")
            print("Starting training from initial configuration")
