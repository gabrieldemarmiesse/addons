# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tfa.seq2seq.seq2seq.beam_search_decoder."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.seq2seq import attention_wrapper
from tensorflow_addons.seq2seq import beam_search_decoder, gather_tree


def test_gather_tree():
    # (max_time = 3, batch_size = 2, beam_width = 3)

    # create (batch_size, max_time, beam_width) matrix and transpose it
    predicted_ids = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]],
        dtype=np.int32,
    ).transpose([1, 0, 2])
    parent_ids = np.array(
        [[[0, 0, 0], [0, 1, 1], [2, 1, 2]], [[0, 0, 0], [1, 2, 0], [2, 1, 1]]],
        dtype=np.int32,
    ).transpose([1, 0, 2])

    # sequence_lengths is shaped (batch_size = 3)
    max_sequence_lengths = [3, 3]

    expected_result = np.array(
        [[[2, 2, 2], [6, 5, 6], [7, 8, 9]], [[2, 4, 4], [7, 6, 6], [8, 9, 10]]]
    ).transpose([1, 0, 2])

    res = gather_tree(
        predicted_ids,
        parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=11,
    )

    np.testing.assert_equal(expected_result, res)


@pytest.mark.parametrize(
    "depth_ndims, merged_batch_beam", [(0, False), (1, False), (1, True), (2, False)]
)
def test_gather_tree_from_array(depth_ndims, merged_batch_beam):
    array = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
            [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 0]],
        ]
    ).transpose([1, 0, 2])
    parent_ids = np.array(
        [
            [[0, 0, 0], [0, 1, 1], [2, 1, 2], [-1, -1, -1]],
            [[0, 0, 0], [1, 1, 0], [2, 0, 1], [0, 1, 0]],
        ]
    ).transpose([1, 0, 2])
    expected_array = np.array(
        [
            [[2, 2, 2], [6, 5, 6], [7, 8, 9], [0, 0, 0]],
            [[2, 3, 2], [7, 5, 7], [8, 9, 8], [11, 12, 0]],
        ]
    ).transpose([1, 0, 2])
    sequence_length = [[3, 3, 3], [4, 4, 3]]

    array = tf.convert_to_tensor(array, dtype=tf.float32)
    parent_ids = tf.convert_to_tensor(parent_ids, dtype=tf.int32)
    expected_array = tf.convert_to_tensor(expected_array, dtype=tf.float32)

    max_time = tf.shape(array)[0]
    batch_size = tf.shape(array)[1]
    beam_width = tf.shape(array)[2]

    def _tile_in_depth(tensor):
        # Generate higher rank tensors by concatenating tensor and
        # tensor + 1.
        for _ in range(depth_ndims):
            tensor = tf.stack([tensor, tensor + 1], -1)
        return tensor

    if merged_batch_beam:
        array = tf.reshape(array, [max_time, batch_size * beam_width])
        expected_array = tf.reshape(expected_array, [max_time, batch_size * beam_width])

    if depth_ndims > 0:
        array = _tile_in_depth(array)
        expected_array = _tile_in_depth(expected_array)

    sorted_array = beam_search_decoder.gather_tree_from_array(
        array, parent_ids, sequence_length
    )

    np.testing.assert_equal(expected_array.numpy(), sorted_array.numpy())


def test_gather_tree_from_array_complex_trajectory():
    # Max. time = 7, batch = 1, beam = 5.
    array = np.expand_dims(
        np.array(
            [
                [[25, 12, 114, 89, 97]],
                [[9, 91, 64, 11, 162]],
                [[34, 34, 34, 34, 34]],
                [[2, 4, 2, 2, 4]],
                [[2, 3, 6, 2, 2]],
                [[2, 2, 2, 3, 2]],
                [[2, 2, 2, 2, 2]],
            ]
        ),
        -1,
    )
    parent_ids = np.array(
        [
            [[0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0]],
            [[0, 1, 2, 3, 4]],
            [[0, 0, 1, 2, 1]],
            [[0, 1, 1, 2, 3]],
            [[0, 1, 3, 1, 2]],
            [[0, 1, 2, 3, 4]],
        ]
    )
    expected_array = np.expand_dims(
        np.array(
            [
                [[25, 25, 25, 25, 25]],
                [[9, 9, 91, 9, 9]],
                [[34, 34, 34, 34, 34]],
                [[2, 4, 2, 4, 4]],
                [[2, 3, 6, 3, 6]],
                [[2, 2, 2, 3, 2]],
                [[2, 2, 2, 2, 2]],
            ]
        ),
        -1,
    )
    sequence_length = [[4, 6, 4, 7, 6]]

    array = tf.convert_to_tensor(array, dtype=tf.float32)
    parent_ids = tf.convert_to_tensor(parent_ids, dtype=tf.int32)
    expected_array = tf.convert_to_tensor(expected_array, dtype=tf.float32)

    sorted_array = beam_search_decoder.gather_tree_from_array(
        array, parent_ids, sequence_length
    )

    np.testing.assert_equal(expected_array.numpy(), sorted_array.numpy())


class TestArrayShapeChecks(tf.test.TestCase):
    def _test_array_shape_dynamic_checks(
        self, static_shape, dynamic_shape, batch_size, beam_width, is_valid=True
    ):
        t = tf.compat.v1.placeholder_with_default(
            np.random.randn(*static_shape).astype(np.float32), shape=dynamic_shape
        )

        batch_size = tf.constant(batch_size)

        def _test_body():
            if tf.executing_eagerly():
                beam_search_decoder._check_batch_beam(t, batch_size, beam_width)
            else:
                with self.cached_session():
                    check_op = beam_search_decoder._check_batch_beam(
                        t, batch_size, beam_width
                    )
                    self.evaluate(check_op)

        if is_valid:
            _test_body()
        else:
            with self.assertRaises(tf.errors.InvalidArgumentError):
                _test_body()

    def test_array_shape_dynamic_checks(self):
        self._test_array_shape_dynamic_checks(
            (8, 4, 5, 10), (None, None, 5, 10), 4, 5, is_valid=True
        )
        self._test_array_shape_dynamic_checks(
            (8, 20, 10), (None, None, 10), 4, 5, is_valid=True
        )
        self._test_array_shape_dynamic_checks(
            (8, 21, 10), (None, None, 10), 4, 5, is_valid=False
        )
        self._test_array_shape_dynamic_checks(
            (8, 4, 6, 10), (None, None, None, 10), 4, 5, is_valid=False
        )
        self._test_array_shape_dynamic_checks(
            (8, 4), (None, None), 4, 5, is_valid=False
        )

    def test_array_shape_static_checks(self):
        self.assertTrue(
            beam_search_decoder._check_static_batch_beam_maybe(
                tf.TensorShape([None, None, None]), 3, 5
            )
        )
        self.assertTrue(
            beam_search_decoder._check_static_batch_beam_maybe(
                tf.TensorShape([15, None, None]), 3, 5
            )
        )
        self.assertFalse(
            beam_search_decoder._check_static_batch_beam_maybe(
                tf.TensorShape([16, None, None]), 3, 5
            )
        )
        self.assertTrue(
            beam_search_decoder._check_static_batch_beam_maybe(
                tf.TensorShape([3, 5, None]), 3, 5
            )
        )
        self.assertFalse(
            beam_search_decoder._check_static_batch_beam_maybe(
                tf.TensorShape([3, 6, None]), 3, 5
            )
        )
        self.assertFalse(
            beam_search_decoder._check_static_batch_beam_maybe(
                tf.TensorShape([5, 3, None]), 3, 5
            )
        )


def test_eos_masking():
    probs = tf.constant(
        [
            [
                [-0.2, -0.2, -0.2, -0.2, -0.2],
                [-0.3, -0.3, -0.3, 3, 0],
                [5, 6, 0, 0, 0],
            ],
            [[-0.2, -0.2, -0.2, -0.2, 0], [-0.3, -0.3, -0.1, 3, 0], [5, 6, 3, 0, 0],],
        ]
    )

    eos_token = 0
    previously_finished = np.array([[0, 1, 0], [0, 1, 1]], dtype=bool)
    masked = beam_search_decoder._mask_probs(probs, eos_token, previously_finished)
    masked = masked.numpy()

    np.testing.assert_equal(probs[0][0], masked[0][0])
    np.testing.assert_equal(probs[0][2], masked[0][2])
    np.testing.assert_equal(probs[1][0], masked[1][0])

    np.testing.assert_equal(masked[0][1][0], 0)
    np.testing.assert_equal(masked[1][1][0], 0)
    np.testing.assert_equal(masked[1][2][0], 0)

    for i in range(1, 5):
        np.testing.assert_allclose(masked[0][1][i], np.finfo("float32").min)
        np.testing.assert_allclose(masked[1][1][i], np.finfo("float32").min)
        np.testing.assert_allclose(masked[1][2][i], np.finfo("float32").min)


def test_step():
    batch_size = 2
    beam_width = 3
    vocab_size = 5
    end_token = 0
    length_penalty_weight = 0.6
    coverage_penalty_weight = 0.0
    dummy_cell_state = tf.zeros([batch_size, beam_width])
    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=tf.nn.log_softmax(tf.ones([batch_size, beam_width])),
        lengths=tf.constant(2, shape=[batch_size, beam_width], dtype=tf.int64),
        finished=tf.zeros([batch_size, beam_width], dtype=tf.bool),
        accumulated_attention_probs=(),
    )

    logits_ = np.full([batch_size, beam_width, vocab_size], 0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 2.7
    logits_[1, 2, 2] = 10.0
    logits_[1, 2, 3] = 0.2
    logits = tf.convert_to_tensor(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=tf.convert_to_tensor(batch_size),
        beam_width=beam_width,
        end_token=end_token,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
    )

    log_probs_ = log_probs.numpy()

    np.testing.assert_equal(outputs.predicted_ids.numpy(), [[3, 3, 2], [2, 2, 1]])
    np.testing.assert_equal(outputs.parent_ids.numpy(), [[1, 0, 0], [2, 1, 0]])
    np.testing.assert_equal(next_beam_state.lengths.numpy(), [[3, 3, 3], [3, 3, 3]])
    np.testing.assert_equal(
        next_beam_state.finished.numpy(), [[False, False, False], [False, False, False]]
    )

    expected_log_probs = []
    expected_log_probs.append(beam_state.log_probs.numpy()[0][[1, 0, 0]])
    expected_log_probs.append(beam_state.log_probs.numpy()[1][[2, 1, 0]])  # 0 --> 1
    expected_log_probs[0][0] += log_probs_[0, 1, 3]
    expected_log_probs[0][1] += log_probs_[0, 0, 3]
    expected_log_probs[0][2] += log_probs_[0, 0, 2]
    expected_log_probs[1][0] += log_probs_[1, 2, 2]
    expected_log_probs[1][1] += log_probs_[1, 1, 2]
    expected_log_probs[1][2] += log_probs_[1, 0, 1]
    np.testing.assert_equal(next_beam_state.log_probs.numpy(), expected_log_probs)


def test_step_with_eos():
    batch_size = 2
    beam_width = 3
    vocab_size = 5
    end_token = 0
    length_penalty_weight = 0.6
    coverage_penalty_weight = 0.0
    dummy_cell_state = tf.zeros([batch_size, beam_width])
    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=tf.nn.log_softmax(tf.ones([batch_size, beam_width])),
        lengths=tf.convert_to_tensor([[2, 1, 2], [2, 2, 1]], dtype=tf.int64),
        finished=tf.convert_to_tensor(
            [[False, True, False], [False, False, True]], dtype=tf.bool
        ),
        accumulated_attention_probs=(),
    )

    logits_ = np.full([batch_size, beam_width, vocab_size], 0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 5.7  # why does this not work when it's 2.7?
    logits_[1, 2, 2] = 1.0
    logits_[1, 2, 3] = 0.2
    logits = tf.convert_to_tensor(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=tf.convert_to_tensor(batch_size),
        beam_width=beam_width,
        end_token=end_token,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
    )

    log_probs_ = log_probs.numpy()

    np.testing.assert_equal(outputs.parent_ids.numpy(), [[1, 0, 0], [1, 2, 0]])
    np.testing.assert_equal(outputs.predicted_ids.numpy(), [[0, 3, 2], [2, 0, 1]])
    np.testing.assert_equal(next_beam_state.lengths.numpy(), [[1, 3, 3], [3, 1, 3]])
    np.testing.assert_equal(
        next_beam_state.finished.numpy(), [[True, False, False], [False, True, False]]
    )

    expected_log_probs = []
    expected_log_probs.append(beam_state.log_probs.numpy()[0][[1, 0, 0]])
    expected_log_probs.append(beam_state.log_probs.numpy()[1][[1, 2, 0]])
    expected_log_probs[0][1] += log_probs_[0, 0, 3]
    expected_log_probs[0][2] += log_probs_[0, 0, 2]
    expected_log_probs[1][0] += log_probs_[1, 1, 2]
    expected_log_probs[1][2] += log_probs_[1, 0, 1]
    np.testing.assert_equal(next_beam_state.log_probs.numpy(), expected_log_probs)


def test_step_large_beam():
    """Tests large beam step.

    Tests a single step of beam search in such case that beam size is
    larger than vocabulary size.
    """
    batch_size = 2
    beam_width = 8
    vocab_size = 5
    end_token = 0
    length_penalty_weight = 0.6
    coverage_penalty_weight = 0.0

    def get_probs():
        """this simulates the initialize method in BeamSearchDecoder."""
        log_prob_mask = tf.one_hot(
            tf.zeros([batch_size], dtype=tf.int32),
            depth=beam_width,
            on_value=True,
            off_value=False,
            dtype=tf.bool,
        )

        log_prob_zeros = tf.zeros([batch_size, beam_width], dtype=tf.float32)
        log_prob_neg_inf = tf.ones([batch_size, beam_width], dtype=tf.float32) * -np.Inf

        log_probs = tf.where(log_prob_mask, log_prob_zeros, log_prob_neg_inf)
        return log_probs

    log_probs = get_probs()
    dummy_cell_state = tf.zeros([batch_size, beam_width])

    _finished = tf.one_hot(
        tf.zeros([batch_size], dtype=tf.int32),
        depth=beam_width,
        on_value=False,
        off_value=True,
        dtype=tf.bool,
    )
    _lengths = np.zeros([batch_size, beam_width], dtype=np.int64)
    _lengths[:, 0] = 2
    _lengths = tf.constant(_lengths, dtype=tf.int64)

    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=log_probs,
        lengths=_lengths,
        finished=_finished,
        accumulated_attention_probs=(),
    )

    logits_ = np.full([batch_size, beam_width, vocab_size], 0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 2.7
    logits_[1, 2, 2] = 10.0
    logits_[1, 2, 3] = 0.2
    logits = tf.constant(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=tf.convert_to_tensor(batch_size),
        beam_width=beam_width,
        end_token=end_token,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
    )

    assert outputs.predicted_ids[0, 0] == 3
    assert outputs.predicted_ids[0, 1] == 2
    assert outputs.predicted_ids[1, 0] == 1
    neg_inf = -np.Inf
    np.testing.assert_equal(
        next_beam_state.log_probs[:, -3:],
        np.array([[neg_inf, neg_inf, neg_inf], [neg_inf, neg_inf, neg_inf]]),
    )
    assert (next_beam_state.log_probs[:, :-3] > neg_inf).numpy().all()
    assert (next_beam_state.lengths[:, :-3] > 0).numpy().all()
    np.testing.assert_equal(
        next_beam_state.lengths[:, -3:], np.array([[0, 0, 0], [0, 0, 0]])
    )


@pytest.mark.parametrize(
    "has_attention, with_alignment_history",
    [(False, False), (True, False), (True, True)],
)
def test_dynamic_decode_rnn(has_attention, with_alignment_history):
    encoder_sequence_length = np.array([3, 2, 3, 1, 1])
    decoder_sequence_length = np.array([2, 0, 1, 2, 3])
    batch_size = 5
    decoder_max_time = 4
    input_depth = 7
    cell_depth = 9
    attention_depth = 6
    vocab_size = 20
    end_token = vocab_size - 1
    start_token = 0
    embedding_dim = 50
    max_out = max(decoder_sequence_length)
    output_layer = tf.keras.layers.Dense(vocab_size, use_bias=True, activation=None)
    beam_width = 3

    batch_size_tensor = tf.constant(batch_size)
    embedding = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    cell = tf.keras.layers.LSTMCell(cell_depth)
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    coverage_penalty_weight = 0.0
    if has_attention:
        coverage_penalty_weight = 0.2
        inputs = tf.compat.v1.placeholder_with_default(
            np.random.randn(batch_size, decoder_max_time, input_depth).astype(
                np.float32
            ),
            shape=(None, None, input_depth),
        )
        tiled_inputs = beam_search_decoder.tile_batch(inputs, multiplier=beam_width)
        tiled_sequence_length = beam_search_decoder.tile_batch(
            encoder_sequence_length, multiplier=beam_width
        )
        attention_mechanism = attention_wrapper.BahdanauAttention(
            units=attention_depth,
            memory=tiled_inputs,
            memory_sequence_length=tiled_sequence_length,
        )
        initial_state = beam_search_decoder.tile_batch(
            initial_state, multiplier=beam_width
        )
        cell = attention_wrapper.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_depth,
            alignment_history=with_alignment_history,
        )
    cell_state = cell.get_initial_state(
        batch_size=batch_size_tensor * beam_width, dtype=tf.float32
    )
    if has_attention:
        cell_state = cell_state.clone(cell_state=initial_state)
    bsd = beam_search_decoder.BeamSearchDecoder(
        cell=cell,
        beam_width=beam_width,
        output_layer=output_layer,
        length_penalty_weight=0.0,
        coverage_penalty_weight=coverage_penalty_weight,
        output_time_major=False,
        maximum_iterations=max_out,
    )

    final_outputs, final_state, final_sequence_lengths = bsd(
        embedding,
        start_tokens=tf.fill([batch_size_tensor], start_token),
        end_token=end_token,
        initial_state=cell_state,
    )

    assert isinstance(final_outputs, beam_search_decoder.FinalBeamSearchDecoderOutput)
    assert isinstance(final_state, beam_search_decoder.BeamSearchDecoderState)

    beam_search_decoder_output = final_outputs.beam_search_decoder_output
    expected_seq_length = 3
    assert (batch_size, expected_seq_length, beam_width) == tuple(
        beam_search_decoder_output.scores.get_shape().as_list()
    )
    assert (batch_size, expected_seq_length, beam_width) == tuple(
        final_outputs.predicted_ids.get_shape().as_list()
    )

    max_sequence_length = np.max(final_sequence_lengths)

    # A smoke test
    assert (
        batch_size,
        max_sequence_length,
        beam_width,
    ) == final_outputs.beam_search_decoder_output.scores.shape
    assert (
        batch_size,
        max_sequence_length,
        beam_width,
    ) == final_outputs.beam_search_decoder_output.predicted_ids.shape
