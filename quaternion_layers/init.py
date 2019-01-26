#!/usr/bin/env python

import numpy as np
from numpy.random import RandomState
import keras.backend as K
from keras import initializers
from keras.initializers import Initializer



class QInit(Initializer):

    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='he', seed=None):


        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion

    def __call__(self, shape, dtype=None):

        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = initializers._compute_fans(
            tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        )

        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        rng = RandomState(1337)

        modulus = rng.uniform(low=-np.sqrt(s)*np.sqrt(3), high=np.sqrt(s)*np.sqrt(3), size=kernel_shape)
        
        phase = rng.uniform(low=-np.pi/2, high=np.pi/2, size=kernel_shape)

        wm = modulus
        wp = phase
        weight = np.concatenate([wp, wm], axis=-1)

        return weight
