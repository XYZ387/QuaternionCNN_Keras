#!/usr/bin/env python

import tensorflow as tf
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
import numpy as np


class QDense(Layer):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(QDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):

        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 3
        data_format = K.image_data_format()
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = initializers._compute_fans(
            kernel_shape,
            data_format=data_format
        )
        s = np.sqrt(1. / fan_in)
        
        def init_phase(shape, dtype=None):
            return np.random.normal(
                size=kernel_shape,
                loc=0,
                scale=np.pi/2,
            )

        def init_modulus(shape, dtype=None):
            return np.random.normal(
                size=kernel_shape,
                loc=0,
                scale=s
            )

        phase_init = init_phase
        modulus_init = init_modulus

        self.phase_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=phase_init,
            name='phase_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.modulus_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=modulus_init,
            name='modulus_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(3 * self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 3 * input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        input_dim = input_shape[-1] // 3
        phase_input = inputs[:, :input_dim]
        modulus_input = inputs[:, input_dim:]

        f_phase   = self.phase_kernel
        f_phase1 = tf.cos(f_phase)
        f_phase2 = tf.sin(f_phase)*(3**0.5/3)
        f_modulus   = self.modulus_kernel

        f1 = (K.pow(f_phase1,2)-K.pow(f_phase2,2))*f_modulus
        f2 = (2*(K.pow(f_phase2,2)-f_phase2*f_phase1))*f_modulus
        f3 = (2*(K.pow(f_phase2,2)+f_phase2*f_phase1))*f_modulus
        f4 = (2*(K.pow(f_phase2,2)+f_phase2*f_phase1))*f_modulus
        f5 = (K.pow(f_phase1,2)-K.pow(f_phase2,2))*f_modulus
        f6 = (2*(K.pow(f_phase2,2)-f_phase2*f_phase1))*f_modulus
        f7 = (2*(K.pow(f_phase2,2)-f_phase2*f_phase1))*f_modulus
        f8 = (2*(K.pow(f_phase2,2)+f_phase2*f_phase1))*f_modulus
        f9 = (K.pow(f_phase1,2)-K.pow(f_phase2,2))*f_modulus
        
        matrix1 = K.concatenate([f1, f2, f3], axis=-1)
        matrix2 = K.concatenate([f4, f5, f6], axis=-1)
        matrix3 = K.concatenate([f7, f8, f9], axis=-1)
        matrix = K.concatenate([matrix1, matrix2, matrix3], axis=0)

        output = K.dot(inputs, matrix)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 3 * self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(QDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

