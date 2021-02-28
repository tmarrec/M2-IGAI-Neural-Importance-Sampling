import numpy as np
from modules import couplings, integrator
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.utils import shuffle, resample

tfd = tfp.distributions  # pylint: disable=invalid-name
tfb = tfp.bijectors  # pylint: disable=invalid-name


def build(in_features, out_features, options):
    """ Builds a dense NN.

    The output layer is initialized to 0, so the first pass
    before training gives the identity transformation.

    Arguments:
        in_features (int): dimensionality of the inputs space
        out_features (int): dimensionality of the output space
        options: additional arguments, not used at the moment

    Returns:
        A tf.keras.models.Model instance

    """
    del options

    invals = tf.keras.layers.Input(in_features, dtype=tf.float64)
    hidden = tf.keras.layers.Dense(32, activation='relu')(invals)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_features, bias_initializer='zeros',
                                    kernel_initializer='zeros')(hidden)
    model = tf.keras.models.Model(invals, outputs)

    return model


class Nice:
    """" Class implementing a NICE (Non-linear Independent Components Estimation) neural network
    to use for Neural Importance Sampling.

    Args:
        -num_coupling_layers: Number of coupling layers to use
        -path_length: Number of bounces to be considered (generates samples of size path_length*2)
        -nb_epoch: Number of loop on given paths to learn
        -kwargs: set different parameters for the nn (silent, num_bins, blob, loss function)
    """

    def __init__(self, num_coupling_layers, path_length, nb_epoch=1, **kwargs):
        """" Initialize the NICE neural network"""
        self.coupling_layers = None
        self.num_coupling_layers = num_coupling_layers
        self.path_length = path_length
        self.nb_epoch = nb_epoch

        self.silent = kwargs.get("silent", True)
        self.num_bins = kwargs.get("num_bins", 32)
        self.blob = kwargs.get("blob", 32)
        self.loss = kwargs.get("loss", "kl")
        self.recomp = kwargs.get("recomp","quadratic")

        # Create the coupling layers
        self.masks = self.make_masks()
        bijector = []

        if self.recomp == "linear":
            for mask in self.masks:
                bijector.append(
                    couplings.PiecewiseLinear(mask, build, num_bins=self.num_bins, blob=self.blob, options=None))
        else :
            for mask in self.masks:
                bijector.append(
                    couplings.PiecewiseQuadratic(mask, build, num_bins=self.num_bins, blob=self.blob, options=None))


        bijector = tfb.Chain(list(reversed(bijector)))

        # Create the tensorflow distribution
        low = np.zeros(path_length * 2, dtype=np.float32)
        high = np.ones(path_length * 2, dtype=np.float32)
        dist = tfd.Uniform(low=low, high=high)
        dist = tfd.Independent(distribution=dist,
                               reinterpreted_batch_ndims=1)
        dist = tfd.TransformedDistribution(
            distribution=dist,
            bijector=bijector)

        # Create the integrator
        optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=10.0)
        self.integrate = integrator.Integrator(dist, optimizer, loss_func=self.loss)

    def make_masks(self):
        """"Create the binary masks according to Neural importance sampling (alternate even and odd dimensions)"""
        base = np.array([1, 0] * self.path_length)

        res = []
        for i in range(self.num_coupling_layers):
            base = (1 - base)
            res.append(base)

        return np.array(res)

    def learn_one(self, paths, probas):
        """"Perform one learning step"""
        return self.integrate.train_one_step(paths, probas)

    def learn(self, paths, probas):
        """"Perform learning on given data"""
        loss = []
        sample_lenght = max(4,((self.nb_epoch//4)+1))


        if not self.silent:
            print("NICE : begin learning with", len(paths), "samples,", self.nb_epoch, "epochs")
        for i in range(self.nb_epoch):
            paths_learn, probas_learn = resample(paths, probas, replace=False, n_samples=(len(paths)//sample_lenght))
            loss.append(self.learn_one(paths_learn, probas_learn))
            if not self.silent:
                print("NICE : epoch n", i, "done")

        return loss

    def generate_paths(self, num_path):
        """Generate a numpy array containing samples according to the learned distribution, and their probabilities"""
        paths = self.integrate.sample(num_path).numpy()

        return paths, self.integrate.sample_weights(paths).numpy()

    def paths_probas(self, paths):
        """"Estimate the probabilities of the given paths"""
        return self.integrate.sample_weights(paths)

    def __str__(self):
        """"Get NICE infos"""

        string = "--- NICE Network for Neural Importance Sampling ---\n"
        string += "Parameters:\n"
        string += "\t-Number of coupling layers:\t" + str(self.num_coupling_layers) + "\n"
        string += "\t-Dimension of samples:\t\t" + str(self.path_length*2) + "\n"
        string += "\t-Epochs on learning data:\t" + str(self.nb_epoch) + "\n"
        string += "\t-Number of bins:\t\t"+str(self.num_bins)+"\n"
        string += "\t-Number of blobs:\t\t"+str(self.blob)+"\n"
        string += "\t-Loss function:\t\t\t"+str(self.loss)+"\n"
        string += "---------------------------------------------------\n"

        return string
