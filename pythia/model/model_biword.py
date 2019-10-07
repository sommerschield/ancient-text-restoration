"""
Copyright 2019 Google LLC, Thea Sommerschield, Jonathan Prag

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections

import sonnet as snt
import tensorflow as tf
from tensor2tensor.models import transformer
from tensor2tensor.utils.learning_rate import learning_rate_schedule


class Model(snt.AbstractModule):

  def __init__(self,
               config,
               alphabet,
               name="model_biword"):
    super(Model, self).__init__(name=name)
    self._config = config
    self._vocab_size_char = len(alphabet.idx2char)
    self._vocab_size_word = len(alphabet.idx2word)
    self._pad_idx = alphabet.pad_idx
    self._unk_idx = alphabet.unk_idx
    self._sos_idx = alphabet.sos_idx
    self._eos_idx = alphabet.eos_idx

  @staticmethod
  def get_learning_rate():
    hparams = transformer.transformer_base()
    return learning_rate_schedule(hparams)

  def _build(self,
             batch,
             keep_prob=0.8,
             sampling_probability=0.5,
             beam_width=0,
             is_training=False):

    # encoder
    with tf.variable_scope('encoder'):
      encoder_output, encoder_state = self._encoder(
        inputs=(tf.transpose(batch['x'], [1, 0]), tf.transpose(batch['x_word'], [1, 0])),
        lengths=batch['x_len'],
        target_size=(self._vocab_size_char, self._vocab_size_word),
        rnn_size_enc=512,
        rnn_size_dec=512,
        rnn_layers_enc=2,
        keep_prob=keep_prob,
        is_training=is_training)

    # decoder
    with tf.variable_scope('decoder'):
      decoder_output = self._seq2seq(encoder_output=encoder_output,
                                     encoder_state=encoder_state,
                                     encoder_lengths=batch['x_len'],
                                     target=batch['y'],
                                     target_lengths=batch['y_len'],
                                     target_size=self._vocab_size_char,
                                     sos_idx=self._sos_idx,
                                     eos_idx=self._eos_idx,
                                     rnn_size_dec=512,
                                     rnn_layers_dec=2,
                                     attention_fn=tf.contrib.seq2seq.LuongAttention,
                                     beam_width=beam_width,
                                     sampling_probability=sampling_probability,
                                     keep_prob=keep_prob,
                                     is_training=is_training)
    return decoder_output

  def _encoder(self,
               inputs,
               lengths,
               target_size,
               rnn_cell=tf.contrib.rnn.LSTMBlockCell,
               rnn_size_enc=128,
               rnn_size_dec=128,
               rnn_layers_enc=3,
               keep_prob=1.,
               is_training=False):

    with tf.variable_scope('encoder'):
      # character embedding
      self.embedding_encoder_char = snt.Embed(target_size[0], embed_dim=rnn_size_enc, name='embedding_encoder_char')
      inputs_char_emb = self.embedding_encoder_char(inputs[0])

      # word embedding
      self.embedding_encoder_word = snt.Embed(target_size[1], embed_dim=rnn_size_enc, name='embedding_encoder_word')
      inputs_word_emb = self.embedding_encoder_word(inputs[1])

      # combine embeddings
      inputs_emb = tf.concat([inputs_char_emb, inputs_word_emb], axis=-1)

      # rnn cells
      cells_fw = [tf.nn.rnn_cell.DropoutWrapper(
        rnn_cell(rnn_size_enc, reuse=tf.AUTO_REUSE,
                 name='rnn_fw_%d' % l),
        input_keep_prob=keep_prob if is_training else 1.,
        dtype=tf.float32) for l in range(rnn_layers_enc)]

      cells_bw = [tf.nn.rnn_cell.DropoutWrapper(
        rnn_cell(rnn_size_enc, reuse=tf.AUTO_REUSE,
                 name='rnn_bw_%d' % l),
        input_keep_prob=keep_prob if is_training else 1.,
        dtype=tf.float32) for l in range(rnn_layers_enc)]

      # run bidirectional rnn
      encoder_output, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=inputs_emb,
        sequence_length=lengths,
        time_major=True,
        dtype=tf.float32)

      # concatenate state of last layer
      encoder_states = []
      for l in range(rnn_layers_enc):
        state_c = tf.concat(
          values=(states_fw[l].c, states_bw[l].c),
          axis=1)
        state_c_bridge = tf.layers.dense(state_c, rnn_size_dec,
                                         trainable=is_training,
                                         name='state_c_bridge_%d' % l,
                                         reuse=tf.AUTO_REUSE)
        state_h = tf.concat(
          values=(states_fw[l].h, states_bw[l].h),
          axis=1)
        state_h_bridge = tf.layers.dense(state_h, rnn_size_dec,
                                         trainable=is_training,
                                         name='state_h_bridge_%d' % l,
                                         reuse=tf.AUTO_REUSE)
        encoder_states.append(tf.contrib.rnn.LSTMStateTuple(c=state_c_bridge, h=state_h_bridge))

    return tf.transpose(encoder_output, [1, 0, 2]), tuple(encoder_states)

  def _seq2seq(self,
               encoder_output,
               encoder_state,
               encoder_lengths,
               target,
               target_lengths,
               target_size,
               sos_idx,
               eos_idx,
               rnn_cell=tf.contrib.rnn.LSTMBlockCell,
               rnn_size_dec=128,
               rnn_layers_dec=3,
               attention_fn=tf.contrib.seq2seq.LuongAttention,
               beam_width=0,
               sampling_probability=0.,
               keep_prob=1.,
               is_training=False):

    with tf.variable_scope('decoder'):
      batch_size = tf.shape(encoder_output)[0]

      # decoder embedding
      self.embedding_decoder = snt.Embed(target_size, embed_dim=rnn_size_dec, name='embedding_decoder')
      self.sos_idx = sos_idx
      self.eos_idx = eos_idx
      self.sos_tokens = tf.fill([batch_size], self.sos_idx)

      # beam search
      if not is_training and beam_width > 0:
        encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output,
                                                       multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,
                                                      multiplier=beam_width)
        encoder_lengths = tf.contrib.seq2seq.tile_batch(encoder_lengths,
                                                        multiplier=beam_width)

      # define attetnion
      self.attention_mechanism = attention_fn(rnn_size_dec,
                                              memory=encoder_output,
                                              memory_sequence_length=encoder_lengths,
                                              scale=True)

      # attention cell
      attention_cell = tf.contrib.rnn.MultiRNNCell([
        tf.nn.rnn_cell.DropoutWrapper(rnn_cell(rnn_size_dec, reuse=tf.AUTO_REUSE, name='rnn_%d' % l),
                                      input_keep_prob=keep_prob if is_training else 1.,
                                      dtype=tf.float32) for l in range(rnn_layers_dec)])

      # attention wrapper
      self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(attention_cell,
                                                              [self.attention_mechanism] * rnn_layers_dec,
                                                              attention_layer_size=[rnn_size_dec] * rnn_layers_dec,
                                                              alignment_history=(not is_training and beam_width == 0))

      # initial attention state
      if not is_training and beam_width > 0:
        bs = batch_size * beam_width
      else:
        bs = batch_size
      decoder_initial_state = self.decoder_cell.zero_state(bs, tf.float32).clone(
        cell_state=encoder_state)

      # projection layer
      self.projection_layer = tf.layers.Dense(target_size,
                                              use_bias=False,
                                              name='output_projection',
                                              trainable=is_training,
                                              _reuse=tf.AUTO_REUSE)

      # training and inference helpers
      if is_training:
        # left pad to add sos idx
        target_sos = tf.pad(target, [[0, 0], [1, 0]], constant_values=self.sos_idx)

        # helper
        target_emb_input = self.embedding_decoder(target_sos)
        if sampling_probability > 0:
          helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            target_emb_input,
            sequence_length=target_lengths,
            embedding=self.embedding_decoder,
            sampling_probability=sampling_probability,
            time_major=False)
        else:
          helper = tf.contrib.seq2seq.TrainingHelper(target_emb_input,
                                                     sequence_length=target_lengths,
                                                     time_major=False)
        # decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                  helper=helper,
                                                  initial_state=decoder_initial_state,
                                                  output_layer=self.projection_layer)
        maximum_iterations = None
      else:
        # inference
        if beam_width > 0:
          decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=self.embedding_decoder,
            start_tokens=self.sos_tokens,
            end_token=self.eos_idx,
            initial_state=decoder_initial_state,
            beam_width=beam_width,
            output_layer=self.projection_layer)
        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self.embedding_decoder,
            start_tokens=self.sos_tokens,
            end_token=self.eos_idx)

          decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                    helper=helper,
                                                    initial_state=decoder_initial_state,
                                                    output_layer=self.projection_layer)

        maximum_iterations = tf.round(self._config.pred_char_max)

      (final_outputs, final_state, final_sequence_lengths) = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=False,
        maximum_iterations=maximum_iterations,
        swap_memory=True)

      if is_training:
        logits = final_outputs.rnn_output
        sample = final_outputs.sample_id
        alignment_history = tf.no_op()
      else:
        logits = tf.no_op()
        if beam_width > 0:
          sample = final_outputs.predicted_ids
          alignment_history = tf.no_op()
        else:
          sample = final_outputs.sample_id

          alignment_history = []
          for history_array in final_state.alignment_history:
            alignment_history.append(history_array.stack())
          alignment_history = tuple(alignment_history)

      return collections.namedtuple('Outputs', 'logits sample alignment_history')(logits=logits, sample=sample,
                                                                                  alignment_history=alignment_history)
