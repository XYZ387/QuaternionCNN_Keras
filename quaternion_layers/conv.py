#!/usr/bin/env python

import tensorflow as tf
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Lambda, Layer, InputSpec, Convolution1D, Convolution2D, add, multiply, Activation, Input, concatenate
from keras.utils import conv_utils
from keras.models import Model
import numpy as np
from .init import QInit


class QConv(Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(QConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = 'channels_last' if rank == 1 else conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis] // 3
        self.kernel_shape = self.kernel_size + (input_dim , self.filters)

        kern_init = QInit(
            kernel_size=self.kernel_size,
            input_dim=input_dim,
            weight_dim=self.rank,
            nb_filters=self.filters,
        )
        
        self.kernel = self.add_weight(
            self.kernel_shape,
            initializer=kern_init,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable = True
        )

        if self.use_bias:
            bias_shape = (3 * self.filters,)
            self.bias = self.add_weight(
                bias_shape,
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable = True
            )

        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim * 3})
        self.built = True

    def call(self, inputs):
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        input_dim    = K.shape(inputs)[channel_axis] // 3
        if self.rank == 1:
            f_phase   = self.kernel[:, :, :self.filters]
            f_modulus   = self.kernel[:, :, self.filters:]

        elif self.rank == 2:
            f_phase   = self.kernel[:,:, :, :self.filters]
            f_modulus   = self.kernel[:, :, :, self.filters:]

        elif self.rank == 3:
            f_phase   = self.kernel[:,:,:, :, :self.filters]
            f_modulus   = self.kernel[:,:,:, :, self.filters:]

        f_phase1 = tf.cos(f_phase)
        f_phase2 = tf.sin(f_phase)*(3**0.5/3)
        convArgs = {"strides":       self.strides[0]       if self.rank == 1 else self.strides,
                    "padding":       self.padding,
                    "data_format":   self.data_format,
                    "dilation_rate": self.dilation_rate[0] if self.rank == 1 else self.dilation_rate}
        convFunc = {1: K.conv1d,
                    2: K.conv2d,
                    3: K.conv3d}[self.rank]


        f1 = (K.pow(f_phase1,2)-K.pow(f_phase2,2))*f_modulus
        f2 = (2*(K.pow(f_phase2,2)-f_phase2*f_phase1))*f_modulus
        f3 = (2*(K.pow(f_phase2,2)+f_phase2*f_phase1))*f_modulus
        f4 = (2*(K.pow(f_phase2,2)+f_phase2*f_phase1))*f_modulus
        f5 = (K.pow(f_phase1,2)-K.pow(f_phase2,2))*f_modulus
        f6 = (2*(K.pow(f_phase2,2)-f_phase2*f_phase1))*f_modulus
        f7 = (2*(K.pow(f_phase2,2)-f_phase2*f_phase1))*f_modulus
        f8 = (2*(K.pow(f_phase2,2)+f_phase2*f_phase1))*f_modulus
        f9 = (K.pow(f_phase1,2)-K.pow(f_phase2,2))*f_modulus
        
        f1._keras_shape = self.kernel_shape
        f2._keras_shape = self.kernel_shape
        f3._keras_shape = self.kernel_shape
        f4._keras_shape = self.kernel_shape
        f5._keras_shape = self.kernel_shape
        f6._keras_shape = self.kernel_shape
        f7._keras_shape = self.kernel_shape
        f8._keras_shape = self.kernel_shape
        f9._keras_shape = self.kernel_shape
        f_phase1._keras_shape = self.kernel_shape
        f_phase2._keras_shape = self.kernel_shape

        matrix1 = K.concatenate([f1, f2, f3], axis=-2)
        matrix2 = K.concatenate([f4, f5, f6], axis=-2)
        matrix3 = K.concatenate([f7, f8, f9], axis=-2)
        matrix = K.concatenate([matrix1, matrix2, matrix3], axis=-1)
        matrix._keras_shape = self.kernel_size + (3 * input_dim, 3 * self.filters)

        output = convFunc(inputs, matrix, **convArgs)

        if self.use_bias:
            output = K.bias_add(
                output,
                self.bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (3 * self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + (3 * self.filters,) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(QConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QConv1D(QConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        super(QConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        config = super(QConv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config


class QConv2D(QConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        super(QConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        config = super(QConv2D, self).get_config()
        config.pop('rank')
        return config


class QConv3D(QConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        super(QConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        config = super(QConv3D, self).get_config()
        config.pop('rank')
        return config
