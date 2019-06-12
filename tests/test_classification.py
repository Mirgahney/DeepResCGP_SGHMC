import numpy as np
import kernels
from models import ClassificationModel
from unittest import TestCase
from argparse import Namespace


class TestClassification(TestCase):
    def test_simple_two_class(self):
        Y = np.random.randint(0, 2, size=(1000, 1))
        X = (Y + np.random.normal(loc=0.0, scale=0.1, size=Y.shape))

        mean = X.mean()
        std = X.std()
        X = (X - mean) / std

        layer_kernels = [
                kernels.SquaredExponential(X.shape[1], lengthscales=2.0),
                kernels.SquaredExponential(32, lengthscales=2.0)
        ]
        options = Namespace(num_inducing=16, iterations=250, minibatch_size=512, window_size=100,
                num_posterior_samples=25, posterior_sample_spacing=25)
        model = ClassificationModel(kernels=layer_kernels, options=options)

        model.fit(X, Y)
        mean, var = model.predict(X)
        correct = (mean.argmax(axis=1) == Y.ravel()).sum()
        self.assertEqual(correct, 1000)

