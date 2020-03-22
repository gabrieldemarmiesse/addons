# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Orginal implementation from keras_contrib/layers/crf
# ==============================================================================
"""Implementing Conditional Random Field layer."""

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.text.crf import crf_log_likelihood
from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
class CRF(tf.keras.layers.Layer):
    """Linear chain conditional random field (CRF).

    References:
        - [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)
    """

    @typechecked
    def __init__(
        self,
        units: int,
        chain_initializer: types.Initializer = "orthogonal",
        chain_regularizer: types.Regularizer = None,
        chain_constraint: types.Constraint = None,
        use_boundary: bool = True,
        boundary_initializer: types.Initializer = "zeros",
        boundary_regularizer: types.Regularizer = None,
        boundary_constraint: types.Constraint = None,
        use_kernel: bool = True,
        kernel_initializer: types.Initializer = "glorot_uniform",
        kernel_regularizer: types.Regularizer = None,
        kernel_constraint: types.Constraint = None,
        use_bias: bool = True,
        bias_initializer: types.Initializer = "zeros",
        bias_regularizer: types.Regularizer = None,
        bias_constraint: types.Constraint = None,
        activation: types.Activation = "linear",
        **kwargs
    ):
        super().__init__(**kwargs)

        # setup mask supporting flag, used by base class (the Layer)
        # because base class's init method will set it to False unconditionally
        # So this assigned must be executed after call base class's init method
        self.supports_masking = True

        self.units = units  # numbers of tags

        self.use_boundary = use_boundary
        self.use_bias = use_bias
        self.use_kernel = use_kernel

        self.activation = tf.keras.activations.get(activation)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.boundary_initializer = tf.keras.initializers.get(boundary_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.chain_regularizer = tf.keras.regularizers.get(chain_regularizer)
        self.boundary_regularizer = tf.keras.regularizers.get(boundary_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.chain_constraint = tf.keras.constraints.get(chain_constraint)
        self.boundary_constraint = tf.keras.constraints.get(boundary_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        # values will be assigned in method
        self.input_spec = None

        # global variable
        self.chain_kernel = None
        self._dense_layer = None
        self.left_boundary = None
        self.right_boundary = None

    def build(self, input_shape):
        input_shape = tuple(tf.TensorShape(input_shape).as_list())

        # see API docs of InputSpec for more detail
        self.input_spec = [tf.keras.layers.InputSpec(shape=input_shape)]

        self._dense_layer = tf.keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs):
        return self._dense_layer(inputs)

    def get_config(self):
        # used for loading model from disk
        config = {
            "units": self.units,
            "use_boundary": self.use_boundary,
            "use_bias": self.use_bias,
            "use_kernel": self.use_kernel,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "chain_initializer": tf.keras.initializers.serialize(
                self.chain_initializer
            ),
            "boundary_initializer": tf.keras.initializers.serialize(
                self.boundary_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "chain_regularizer": tf.keras.regularizers.serialize(
                self.chain_regularizer
            ),
            "boundary_regularizer": tf.keras.regularizers.serialize(
                self.boundary_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "chain_constraint": tf.keras.constraints.serialize(self.chain_constraint),
            "boundary_constraint": tf.keras.constraints.serialize(
                self.boundary_constraint
            ),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @property
    def _compute_dtype(self):
        # fixed output dtype from underline CRF functions
        return tf.int32


@tf.keras.utils.register_keras_serializable(package="Addons")
class CRFLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        potentials, y_true, sequence_length, chain_kernel = inputs
        y_true = tf.cast(y_true, tf.int32)
        sequence_length = tf.cast(sequence_length, tf.int32)

        log_likelihood, _ = crf_log_likelihood(
            potentials, y_true, sequence_length, chain_kernel
        )
        return -log_likelihood

    def get_config(self):
        return super().get_config()
