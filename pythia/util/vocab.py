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

import editdistance
import numpy as np


def text_to_idx(t, alphabet):
  return np.array([alphabet.char2idx[c] for c in t], dtype=np.int32)


def text_to_word_idx(t, alphabet):
  out = np.full(len(t), alphabet.word2idx[alphabet.unk], dtype=np.int32)

  for m in re.finditer(r'\w+', t):
    if m.group() in alphabet.word2idx:
      out[m.start():m.end()] = alphabet.word2idx[m.group()]

  return out


def idx_to_text(idxs, alphabet):
  idxs = np.array(idxs)
  out = ''
  for i in range(idxs.size):
    idx = idxs[i]
    if idx == alphabet.eos_idx:
      break
    elif idx not in [alphabet.sos_idx]:
      out += alphabet.idx2char[idx]

  return out


def idx_to_text_batch(idxs, alphabet, lengths=None):
  b = []
  for i in range(idxs.shape[0]):
    idxs_i = idxs[i]
    if lengths:
      idxs_i = idxs_i[:lengths[i]]
    b.append(idx_to_text(idxs_i, alphabet))
  return b


def edit_distance_batch(hyp, tar, tar_len, eos_idx):
  cer = 0.
  bs = hyp.shape[0]

  for i in range(bs):
    # filter hyp for eos
    hyp_len = np.argmax(hyp[i] == eos_idx, axis=0)
    if hyp_len.size == 0:
      hyp_len = hyp[i].size

    # filter tar for eos
    eos_pos = np.argmax(tar[i] == eos_idx, axis=0)
    if eos_pos.size > 0:
      tar_len[i] = eos_pos

    cer += editdistance.eval(hyp[i, :hyp_len], tar[i, :tar_len[i]]) / float(tar_len[i])
  return np.float32(cer / bs)
