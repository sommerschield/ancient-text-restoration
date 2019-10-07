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

import re

import numpy as np
import tensorflow as tf

from ..util.vocab import text_to_idx, text_to_word_idx


def generate_sample(config, alphabet, texts, idx=None, eos=False, pad=False, is_training=False):
  iter_count = 0
  x_idx, x_len, x_word_idx, x_word_len, y_idx, y_len = None, None, None, None, None, None

  # select a random text until constraints are satisfied
  while True:
    iter_count += 1

    if idx is None:
      t = np.random.choice(texts)
    else:
      t = texts[idx]
      if iter_count > 1:
        break
    text = ' '.join([s + '.' for s in t.sentences])

    # remove guess signs
    if not config.pred_guess:
      text = text.replace(alphabet.sog, '').replace(alphabet.eog, '')

    # randomly select context size and number of characters to remove
    context_char_max = min(max(config.context_char_min, len(text)), config.context_char_max)
    if config.context_char_min > 0:
      context_char_num = np.random.randint(config.context_char_min, context_char_max)
    else:
      context_char_num = context_char_max

    if len(text.replace(alphabet.sog, '').replace(alphabet.eog, '')) < context_char_num:
      continue

    # compute start sentence
    text_idx_start = np.random.randint(0, len(text) - context_char_num) if len(text) > context_char_num else 0
    text_idx_end = text_idx_start + context_char_num

    # keep only current text
    text = text[text_idx_start:text_idx_end]

    if config.pred_guess and not is_training:
      # delete guess characters
      matches = []
      for m in re.finditer(r'%s([^%s%s]+)%s' % (
          re.escape(alphabet.sog),
          re.escape(alphabet.missing),
          re.escape(alphabet.eog), re.escape(alphabet.eog)), text):
        start = m.start() + 1
        end = m.end() - 2
        if config.pred_char_min <= end - start <= config.pred_char_max:
          matches.append((m.group(1), start, end))

      # skip if no matches found
      if len(matches) == 0:
        continue

      # pick a random match
      matches_idx = np.random.randint(len(matches))
      (y, pred_start, pred_end) = matches[matches_idx]
      x = list(text)
      for i in range(pred_start, pred_end + 1):
        x[i] = alphabet.pred

      # remove guess signs
      x = [c for c in x if c not in [alphabet.sog, alphabet.eog]]

    else:
      # delete pred characters
      if alphabet.pred not in text:
        pred_char_num = np.random.randint(config.pred_char_min, min(len(text), config.pred_char_max))
        if len(text) < pred_char_num:
          continue
        pred_char_idx = np.random.randint(0, len(text) - pred_char_num) if len(text) > pred_char_num else 0
        y = text[pred_char_idx:pred_char_idx + pred_char_num]
      else:
        y = ''

      # skip if it's a missing character
      if alphabet.missing in y:
        continue

      x = list(text)
      if alphabet.pred not in x:
        for i in range(pred_char_idx, pred_char_idx + pred_char_num):
          x[i] = alphabet.pred

      # hide random characters
      if config.missing_char_max > 0 and is_training:
        missing_char_num = np.random.randint(config.missing_char_min, min(len(text), config.missing_char_max))
        for i in np.random.randint(0, context_char_num, missing_char_num):
          if x[i] != alphabet.pred:
            x[i] = alphabet.missing

    # convert to string
    x = ''.join(x)

    # convert to indices
    x_idx = text_to_idx(x, alphabet)
    x_word_idx = text_to_word_idx(x, alphabet)
    y_idx = text_to_idx(y, alphabet)
    assert (len(x_idx) == len(x_word_idx))

    # append eos character
    if eos:
      y_idx = np.concatenate((y_idx, [alphabet.eos_idx]))

    # compute lengths
    x_len = np.int32(x_idx.shape[0])
    x_word_len = np.int32(x_word_idx.shape[0])
    y_len = np.int32(y_idx.shape[0])

    # pad sequences
    if pad:
      x_idx = np.pad(x_idx, (0, config.context_char_max - x_idx.size), 'constant',
                     constant_values=(None, alphabet.eos_idx))
      x_word_idx = np.pad(x_word_idx, (0, config.context_char_max - x_word_idx.size), 'constant',
                          constant_values=(None, alphabet.eos_idx))
      y_idx = np.pad(y_idx, (0, config.pred_char_max - y_idx.size + 1), 'constant',
                     constant_values=(None, alphabet.eos_idx))

    break

  return {'x': x_idx, 'x_len': x_len,
          'x_word': x_word_idx, 'x_word_len': x_word_len,
          'y': y_idx, 'y_len': y_len}


def tf_dataset(config, alphabet, texts, is_training=False):
  with tf.device('/cpu:0'):
    # sample generator
    def generate_samples():
      while True:
        yield generate_sample(config, alphabet, texts, eos=True, pad=False,
                              is_training=is_training)

    # create dataset from generator
    ds = tf.data.Dataset.from_generator(
      generate_samples,
      {'x': tf.int32, 'x_len': tf.int32,
       'x_word': tf.int32, 'x_word_len': tf.int32,
       'y': tf.int32, 'y_len': tf.int32})

    # batch and pad samples
    ds = ds.padded_batch(config.batch_size,
                         padded_shapes={'x': [None], 'x_len': [],
                                        'x_word': [None], 'x_word_len': [],
                                        'y': [None], 'y_len': []},
                         padding_values={'x': alphabet.eos_idx, 'x_len': 0,
                                         'x_word': alphabet.eos_idx, 'x_word_len': 0,
                                         'y': alphabet.eos_idx, 'y_len': 0})

    # repeat
    ds = ds.repeat()

    # enable prefetch
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
    ds_iterator = ds.make_one_shot_iterator()

    return ds_iterator.get_next()
