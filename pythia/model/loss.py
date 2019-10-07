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

import tensorflow as tf

from ..util.vocab import edit_distance_batch


def loss_seq2seq(logits, labels, labels_lengths, smoothing=False):
  def cross_entropy_smoothed(labels, logits, smoothing=0.1):
    with tf.name_scope('cross_entropy_smoothed'):
      num_labels = tf.shape(logits)[1]
      onehot_labels = tf.one_hot(labels, num_labels, dtype=tf.float32)
      if smoothing > 0:
        num_classes = tf.cast(tf.shape(onehot_labels)[1], tf.float32)
        smooth_pos = 1. - smoothing
        smooth_neg = smoothing / num_classes
        onehot_labels = onehot_labels * smooth_pos * smooth_neg

    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits)

  # trim logit length to max time
  max_time = tf.shape(labels)[1]
  logits = logits[:, :max_time, :]

  label_weights = tf.sequence_mask(labels_lengths, max_time, dtype=tf.float32)

  if smoothing:
    loss = tf.contrib.seq2seq.sequence_loss(logits,
                                            labels,
                                            label_weights,
                                            softmax_loss_function=cross_entropy_smoothed,
                                            average_across_timesteps=True,
                                            average_across_batch=True)
  else:
    loss = tf.contrib.seq2seq.sequence_loss(logits,
                                            labels,
                                            label_weights,
                                            average_across_timesteps=True,
                                            average_across_batch=True)
  return tf.reduce_mean(loss)


def error(batch, outputs, alphabet):
  # sample dimensions
  sample_dims = len(outputs.sample.get_shape().as_list())
  if sample_dims == 3:
    sample = outputs.sample[:, :, 0]
  else:
    sample = outputs.sample

  cer = tf.py_func(edit_distance_batch,
                   [sample, batch['y'], batch['y_len'], alphabet.eos_idx],
                   tf.float32)
  return cer
