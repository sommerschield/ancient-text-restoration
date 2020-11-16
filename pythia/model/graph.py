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

from ..data.generator import tf_dataset
from ..model.loss import loss_seq2seq, error


def build_graph(config, alphabet, texts, model):
  # evaluation graph
  def valid_batch_output():
    with tf.name_scope('valid_batch'):
      valid_batch = tf_dataset(config, alphabet, texts[config.dataset_test_set], is_training=False)

    # evaluation model
    with tf.name_scope('valid_model'):
      valid_output = model(valid_batch,
                           keep_prob=1.,
                           sampling_probability=0,
                           beam_width=config.beam_width,
                           is_training=False)
    return valid_batch, valid_output

  # get training step counter.
  global_step = tf.train.get_or_create_global_step()

  # training graph
  with tf.name_scope('train_batch'):
    train_batch = tf_dataset(config, alphabet, texts['train'], is_training=True)

  # training model
  with tf.name_scope('train_model'):
    train_output = model(train_batch,
                         is_training=True)

  with tf.name_scope('train'):
    train_loss = loss_seq2seq(train_output.logits,
                              train_batch['y'],
                              train_batch['y_len'],
                              smoothing=False)
    tf.summary.scalar('train_loss', train_loss, collections=['train'])

  # optimizer
  with tf.name_scope('optimizer'):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(train_loss, tvars), config.grad_clip)
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    train_step = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

  # evaluate performance
  with tf.name_scope('valid'):
    valid_cer = tf.constant(0.)
    valid_count = tf.constant(0)
    valid_iter = int(config.eval_samples / config.batch_size)

    def valid_while_cond(valid_count, _):
      return tf.less(valid_count, valid_iter)

    def valid_while_body(valid_count, valid_cer):
      valid_batch, valid_output = valid_batch_output()

      valid_cer += error(valid_batch, valid_output, alphabet) / float(valid_iter)
      valid_count += 1

      valid_cer.set_shape([])
      valid_count.set_shape([])
      return valid_count, valid_cer

    valid_count, valid_cer = tf.while_loop(valid_while_cond,
                                           valid_while_body,
                                           [valid_count, valid_cer],
                                           back_prop=False,
                                           swap_memory=True)
    tf.summary.scalar('valid_cer', valid_cer, collections=['valid'])

  # sample outputs
  valid_batch, valid_output = valid_batch_output()

  return {
    'train_loss': train_loss,
    'valid_cer': valid_cer,
    'valid_output': valid_output,
    'valid_batch': valid_batch,
    'summary_train': tf.summary.merge_all('train'),
    'summary_valid': tf.summary.merge_all('valid'),
    'global_step': global_step,
    'train_step': train_step,
    'init': tf.global_variables_initializer()
  }
