import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import sys
import math
import numbers
import numpy as np
import random as rd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import settings


def set_seed():
    np.random.seed(settings.SEED)
    rd.seed(settings.SEED)
    tf.random.set_seed(settings.SEED)


class MaskInputs(tf.keras.layers.Layer):
    def __init__(self, masked_value=0):
        super(MaskInputs, self).__init__()
        self.masked_value = masked_value
        if not isinstance(self.masked_value, numbers.Real):
            raise ValueError("masked value must be a scalar (float or int), "
                             f"got instead {self.masked_value}")

    def call(self, inputs):
        return tf.dtypes.cast(inputs==self.masked_value, tf.float32)


class Encoder(tf.keras.layers.Layer):
    """
    Encoder layer
    """
    def __init__(self,
                 input_dim,
                 activation=tf.nn.elu,
                 depth=2,
                 layer_size_decay=settings.HIDDEN_DIM_FRACTION,
                 weight_init=tf.keras.initializers.GlorotUniform(seed=settings.SEED)):
        """
        Initialization of Encoder layer.

        :param input_dim: number of input features
        :type input_dim: int
        :param activation: activation, defaults to tf.nn.elu
        :type activation: tf.keras activation
        :param depth: depth of encoder, defaults to 2
        :type depth: int, optional
        :param layer_size_decay: rate of layers sizes decay with a depth,
                defaults to settings.HIDDEN_DIM_FRACTION
        :type layer_size_decay: float, optional
        :param weight_init: initialization of weights, defaults to tf.keras.initializers.GlorotUniform
        :type weight_init: tf.keras.initializers
        """
        super(Encoder, self).__init__()

        self.activation = activation
        self.depth = depth
        if not isinstance(self.depth, numbers.Real) or self.depth < 1:
            raise ValueError("`depth` must be a positive int (>=1), "
                             f"got instead {self.depth}")

        self.input_dim = input_dim
        self.layer_size_decay = layer_size_decay
        decay_constraints = (self.layer_size_decay > 0) and (self.layer_size_decay < 1)
        if not isinstance(self.layer_size_decay, numbers.Real):
            raise ValueError("`layer_size_decay` must be a positive float (0 < x < 1), "
                             f"got instead {self.layer_size_decay}")

        if not decay_constraints:
            raise ValueError("`layer_size_decay` must be a positive float (0 < x < 1), "
                             f"got instead {self.layer_size_decay}")

        self.weight_init = weight_init

        self.hidden_dimensions = []
        for layer_idx in range(1, depth+1):
            dim = self.input_dim * self.layer_size_decay**layer_idx
            self.hidden_dimensions.append(math.ceil(dim))

        self.output_dim = self.hidden_dimensions[-1]
        if self.output_dim == 1:
            raise ValueError(f"Bottleneck is too shallow (={self.output_dim}). "
                             "Consider larger value of `layer_size_decay` or "
                             "smaller value of `depth`.")

        self.layers = []
        for layer_idx in range(self.depth):
            layer = tf.keras.layers.Dense(self.hidden_dimensions[layer_idx],
                                          activation=self.activation,
                                          kernel_initializer=self.weight_init)
            self.layers.append(layer)

    def call(self, inputs):
        x = inputs
        for layer_idx in range(self.depth):
            x = self.layers[layer_idx](x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 activation=tf.nn.elu,
                 depth=2,
                 layer_size_rise=1/settings.HIDDEN_DIM_FRACTION,
                 weight_init=tf.keras.initializers.GlorotUniform(seed=settings.SEED)):
        super(Decoder, self).__init__()

        self.activation = activation
        self.depth = depth
        if not isinstance(self.depth, numbers.Real) or self.depth < 1:
            raise ValueError("`depth` must be a positive int (>=1), "
                             f"got instead {self.depth}")
        self.input_dim = input_dim
        self.layer_size_rise = layer_size_rise
        rise_constraints = (self.layer_size_rise > 1)
        if not isinstance(self.layer_size_rise, numbers.Real):
            raise ValueError("`layer_size_rise` must be a positive float (>1), "
                             f"got instead {self.layer_size_rise}")
        if not rise_constraints:
            raise ValueError("`layer_size_rise` must be a positive float (>1), "
                             f"got instead {self.layer_size_rise}")

        self.weight_init = weight_init

        self.hidden_dimensions = []
        for layer_idx in range(1, depth+1):
            dim = self.input_dim * self.layer_size_rise**layer_idx
            self.hidden_dimensions.append(math.floor(dim))

        self.output_dim = self.hidden_dimensions[-1]

        self.layers = []
        for layer_idx in range(self.depth):
            layer = tf.keras.layers.Dense(self.hidden_dimensions[layer_idx],
                                          activation=self.activation,
                                          kernel_initializer=self.weight_init)
            self.layers.append(layer)

    def call(self, inputs):
        x = inputs
        for layer_idx in range(self.depth):
            x = self.layers[layer_idx](x)
        return x


class Autoencoder(tf.keras.Model):
    def __init__(self,
                 input_dim,
                 layer_size_decay=settings.HIDDEN_DIM_FRACTION,
                 layer_size_rise=1/settings.HIDDEN_DIM_FRACTION,
                 hidden_activation=tf.nn.selu,
                 # output_activation=tf.nn.sigmoid,
                 output_activation=tf.keras.activations.linear,
                 weight_init=tf.keras.initializers.GlorotUniform(seed=settings.SEED),
                 use_auxiliary_features=True):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.layer_size_decay = layer_size_decay
        self.layer_size_rise = layer_size_rise
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.encoder = Encoder(input_dim=self.input_dim,
                               activation=self.hidden_activation,
                               depth=settings.DEPTH,
                               layer_size_decay=self.layer_size_decay,
                               weight_init=self.weight_init)

        self.decoder = Decoder(input_dim=self.encoder.output_dim,
                               activation=self.hidden_activation,
                               depth=settings.DEPTH,
                               layer_size_rise=self.layer_size_rise,
                               weight_init=self.weight_init)
        self.imputator = tf.keras.layers.Dense(self.input_dim,
                                               activation=self.output_activation,
                                               kernel_initializer=self.weight_init)
        self.maskout = MaskInputs(masked_value=0.0)
        self.use_auxiliary_features = use_auxiliary_features

    def call(self, inputs):
        if self.use_auxiliary_features:
            main_feats, aux_feats = inputs
            x = tf.keras.layers.Concatenate()([main_feats, aux_feats])
        else:
            main_feats, _ = inputs
            x = main_feats

        x = self.encoder(x)
        x = self.decoder(x)
        x = tf.keras.layers.Concatenate()([main_feats, x])
        if self.use_auxiliary_features:
            x = tf.keras.layers.Concatenate()([aux_feats, x])
        x = self.imputator(x)

        mask = self.maskout(main_feats)
        x = tf.keras.layers.Multiply()([x, mask])
        x = tf.keras.layers.Add()([main_feats, x])
        return x


def scheduler(epoch, lr):
    LR_EPS = 1e-6
    return lr * tf.math.abs(tf.math.sin(tf.cast(epoch, tf.float32))) + LR_EPS


def callbacks(save_path="PSW"):
    model_checkpoint_callback = ModelCheckpoint(filepath=f"{save_path}/{settings.CHECKPOINT_PATH}",
                                                save_weights_only=False,
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True)

    early_stopping_callback = EarlyStopping(monitor="val_loss",
                                            patience=10)

    learning_rate_scheduler = ReduceLROnPlateau(monitor="val_loss",
                                                factor=0.1,
                                                min_delta=0.5e-8,
                                                verbose=1,
                                                patience=7)
    learning_rate_sin = LearningRateScheduler(scheduler)
    return [model_checkpoint_callback,
            early_stopping_callback,
            learning_rate_scheduler,
            # learning_rate_sin
            ]
