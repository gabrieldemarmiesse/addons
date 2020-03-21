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
# ==============================================================================
"""Tests for Conditional Random Field loss."""

import itertools
import math
import os
import sys

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.util import nest

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.losses import crf
from tensorflow_addons.utils import test_utils

from unittest.mock import patch

CRF_LOSS_OBJ_LIST = [crf.crf_loss, crf.ConditionalRandomFieldLoss()]


def get_some_crf_layer():
    transitions = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.8, 0.3, 0.1, 0.7, 0.9],
            [-0.3, 2.1, -5.6, 3.4, 4.0],
            [0.2, 0.4, 0.6, -0.3, -0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    boundary_values = np.ones((5,))
    return CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=tf.keras.initializers.Constant(transitions),
        use_boundary=True,
        boundary_initializer=tf.keras.initializers.Constant(boundary_values),
        name="crf_layer",
    )


@test_utils.run_all_in_graph_and_eager_modes
class ConditionalRandomFieldLossTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()

        self.logits = np.array(
            [
                [[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]],
                [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]],
            ]
        )
        self.tags = np.array([[2, 3, 4], [3, 2, 2]])

        self.transitions = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.8, 0.3, 0.1, 0.7, 0.9],
                [-0.3, 2.1, -5.6, 3.4, 4.0],
                [0.2, 0.4, 0.6, -0.3, -0.4],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        self.boundary_values = np.ones((5,))

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=tf.keras.initializers.Constant(self.transitions),
            use_boundary=True,
            boundary_initializer=tf.keras.initializers.Constant(self.boundary_values),
            name="crf_layer",
        )

    def score(self, logits, tags):
        """Computes the likelihood score for the given sequence of tags, given
        the provided logits (and the transition weights in the CRF model)"""
        # Start with transitions from START and to END
        total = self.boundary_values[tags[0]] + self.boundary_values[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    def compute_log_likelihood(self):
        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores
        # for the logits across all possible tags)
        for logits_i, tags_i in zip(self.logits, self.tags):
            numerator = self.score(logits_i, tags_i)
            all_scores = [
                self.score(logits_i, tags_j)
                for tags_j in itertools.product(range(5), repeat=3)
            ]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        return manual_log_likelihood

    def _test_loss_function(self, loss_obj):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile("adam", loss=loss_obj, metrics=[tf.keras.metrics.Accuracy()])

        log_likelihood, _ = model.train_on_batch(self.logits, self.tags)

        # The manually computed log likelihood should
        # equal the result of crf.forward.
        expected_log_likelihood = self.compute_log_likelihood()
        unbatched_log_likelihood = -2 * log_likelihood

        self.assertAllClose(expected_log_likelihood, unbatched_log_likelihood)

    def test_class_loss_function(self):
        self._test_loss_function(crf.ConditionalRandomFieldLoss())

    def test_func_loss_function(self):
        self._test_loss_function(crf.crf_loss)

    def test_model_fit(self):
        for loss_obj in CRF_LOSS_OBJ_LIST:
            with self.subTest(loss_obj=loss_obj):
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Input(shape=(3, 5)))
                model.add(self.crf)
                model.compile(
                    "adam", loss=loss_obj, metrics=[tf.keras.metrics.Accuracy()]
                )

                model.fit(self.logits, self.tags, epochs=10, batch_size=1)

    def _test_dump_and_load(self, loss_obj):
        tmp_dir = self.get_temp_dir()
        MODEL_PERSISTENCE_PATH = os.path.join(tmp_dir, "test_saving_crf_model.h5")

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile("adam", loss=loss_obj, metrics=[tf.keras.metrics.Accuracy()])

        model.fit(self.logits, self.tags, epochs=10, batch_size=1)

        model.save(MODEL_PERSISTENCE_PATH)

        # no news is good news
        new_model = tf.keras.models.load_model(MODEL_PERSISTENCE_PATH)
        new_model.fit(self.logits, self.tags, epochs=10, batch_size=1)

        try:
            os.remove(MODEL_PERSISTENCE_PATH)
        except OSError:
            pass

    def test_dump_and_load_with_class_loss(self):
        # TODO(howl-anderson): wait for the PR merged
        self.skipTest("require tensorflow/tensorflow#37018 merged")

        self._test_dump_and_load(crf.ConditionalRandomFieldLoss())

    def test_mask_left_padding(self):
        for loss_obj in CRF_LOSS_OBJ_LIST:
            with self.subTest(loss_obj=loss_obj):
                train_x = np.array(
                    [
                        [
                            # O   B-X  I-X  B-Y  I-Y
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                        ],
                        [
                            # O   B-X  I-X  B-Y  I-Y
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                )

                train_y = np.array(
                    [[1, 2, 2], [1, 1, 1]]  # B-X  I-X  I-X  # B-X  B-X  B-X
                )

                mask = np.array([[0, 1, 1], [1, 1, 1]])

                layer = CRF(5)

                x = tf.keras.layers.Input(shape=(3, 5))
                y = layer(x, mask=tf.constant(mask))

                # check shape inference
                model = tf.keras.models.Model(x, y)
                model.compile("adam", loss_obj)

                with self.assertRaises(tf.errors.InvalidArgumentError) as context:
                    model.fit(train_x, train_y)

                self.assertTrue(
                    "CRF layer do not support left padding" in context.exception.message
                )


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_mask_right_padding(loss_obj):
    train_x = np.array(
        [
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    train_y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X

    mask = np.array([[1, 1, 1], [1, 1, 0]])

    layer = CRF(5)

    x = tf.keras.layers.Input(shape=(3, 5))
    y = layer(x, mask=tf.constant(mask))

    # check shape inference
    model = tf.keras.models.Model(x, y)
    model.compile("adam", loss_obj)
    model.fit(train_x, train_y)


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_in_subclass_model(loss_obj):

    train_x = np.array(
        [
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    train_y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X

    def patch_mark_as_return(outputs, acd):
        """Marks `outputs` as the return values for automatic control
        deps."""

        def _mark_as_return(tensor):
            """Marks `tensor` as the return value for automatic control
            deps."""
            if not tensor_util.is_tensor(tensor):
                return tensor

            # pylint: disable=protected-access
            return_tensor = acd.mark_as_return(tensor)
            if getattr(tensor, "_keras_mask", None) is not None:
                return_tensor._keras_mask = acd.mark_as_return(tensor._keras_mask)
            else:
                return_tensor._keras_mask = None

            # TODO(howl-anderson) a little hack here, handle _keras_history
            if getattr(tensor, "_keras_history", None) is not None:
                return_tensor._keras_history = tensor._keras_history

            # Handle TensorFlow Probability attached metadata.
            # TODO(b/132076537): Remove this once TFP uses `CompositeTensor`.
            if getattr(tensor, "_tfp_distribution", None) is not None:
                return_tensor._tfp_distribution = tensor._tfp_distribution

            return return_tensor
            # pylint: enable=protected-access

        return nest.map_structure(_mark_as_return, outputs)

    class CRFModel(tf.keras.Model):
        def __init__(self):
            super().__init__()

            self.layer = CRF(5)

        def call(self, inputs):
            return self.layer(inputs)

        @patch.object(base_layer_utils, "mark_as_return", patch_mark_as_return)
        def __call__(self, inputs, *args, **kwargs):
            outputs = super().__call__(inputs, *args, **kwargs)

            # A hack that add _keras_history to EagerTensor, make it more like normal Tensor
            for tensor in tf.nest.flatten(outputs):
                if not hasattr(tensor, "_keras_history"):
                    tensor._keras_history = (self, 0, 0)

            return outputs

    model = CRFModel()

    model.compile("adam", loss_obj)
    model.fit(train_x, train_y)


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_serialization(loss_obj):
    ref_fn = loss_obj
    config = tf.keras.losses.serialize(ref_fn)
    fn = tf.keras.losses.deserialize(config)
    assert ref_fn.get_config() == fn.get_config()


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_keras_model_compile(loss_obj):
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Input(shape=(3, 5)), get_some_crf_layer()]
    )

    model.compile(loss=loss_obj, optimizer="adam")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
