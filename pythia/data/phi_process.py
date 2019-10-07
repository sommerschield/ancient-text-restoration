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

import argparse
import glob
import os
import pickle
import re
from collections import Counter

from tqdm import tqdm

from ..include.alphabet import GreekAlphabet
from ..include.dataset import Dataset
from ..include.text import Text
from ..util.text_clean import text_to_sentences, text_clean_phi, strip_accents
from ..util.text_stats import texts_statistics

p = argparse.ArgumentParser(description='PHI process')
p.add_argument('--input_dir', default=os.getcwd() + '/pythia/data/phi/', type=str,
               help='input PHI path.')
p.add_argument('--output_plaintext_dir', default=os.getcwd() + '/pythia/data/phi-plaintext/',
               type=str,
               help='output PHI plaintext path.')
p.add_argument('--output_dataset_dir', default=os.getcwd() + '/pythia/data/datasets/', type=str,
               help='output PHI processed path.')
FLAGS = p.parse_args()


def process_greek_epigraphy(save=True):
  #######################################
  # Load copra
  #######################################

  dataset = {}
  greek_alphabet = GreekAlphabet(wordlist_path=None)

  #######################################
  # Phi7
  #######################################
  dataset['phi7'] = Dataset()

  os.makedirs(FLAGS.output_plaintext_dir, exist_ok=True)

  pred_char_max = 20
  context_char_min = 100

  # plain text paths
  text_paths = glob.glob(os.path.expanduser(os.path.join(FLAGS.input_dir, '*.txt')))
  for text_path in tqdm(text_paths):
    with open(text_path, 'r') as f:

      # clean text
      t = text_clean_phi(f.read(), greek_alphabet)

      # tokenize sentences
      sentences = text_to_sentences(t, greek_alphabet)

    # append to text
    t = ' '.join([s + '.' for s in sentences])
    if len(sentences) > 0 and len(re.findall(r'\w', t)) > context_char_min:
      text = Text(path=text_path, sentences=sentences)
      dataset['phi7'].texts.append(text)

      # store it to a file
      with open(text_path.replace(FLAGS.input_dir, FLAGS.output_plaintext_dir), 'w') as f_plain:
        f_plain.write(t)

  texts_statistics(dataset['phi7'].texts)

  # Texts
  texts = dataset['phi7'].texts
  print('Texts:', len(texts))

  # split to train, valid, test
  texts_dict = {'train': [], 'valid': [], 'test': []}

  # Find test/valid texts
  count = {'guess_wrong': 0, 'test_val': 0, 'missing_ratio': 0.}
  for text in texts:
    t = ' '.join([s + '.' for s in text.sentences])

    count['missing_ratio'] += t.count('-') / float(len(t)) / len(texts)

    # Check guess signs count
    test_val = False
    if t.count(greek_alphabet.sog) != t.count(greek_alphabet.eog):
      count['guess_wrong'] += 1
    else:
      for m in re.findall(r'%s([^%s%s]+)%s' % (re.escape(greek_alphabet.sog), re.escape(greek_alphabet.missing),
                                               re.escape(greek_alphabet.eog), re.escape(greek_alphabet.eog)), t):
        if 1 <= len(m) <= pred_char_max:
          test_val = True

    # If it can go to test and validation set
    path_lastdigit = int(os.path.splitext(os.path.basename(text.path))[0][-1])
    if test_val and path_lastdigit == 3:
      count['test_val'] += 1
      texts_dict['test'].append(text)
    elif test_val and path_lastdigit == 4:
      count['test_val'] += 1
      texts_dict['valid'].append(text)
    else:
      texts_dict['train'].append(text)

  print(count)

  # statistics
  print('texts_train')
  texts_statistics(texts_dict['train'], frequencies=False)
  print('texts_valid')
  texts_statistics(texts_dict['valid'], frequencies=False)
  print('texts_test')
  texts_statistics(texts_dict['test'], frequencies=False)

  # Save
  if save:
    # Create structure
    os.makedirs(FLAGS.output_dataset_dir, exist_ok=True)

    with open(os.path.join(FLAGS.output_dataset_dir, "greek_epigraphy_dict.p"), "wb") as f:
      pickle.dump(texts_dict, f, -1)

  return texts_dict


def generate_wordlist():
  # Create structure
  os.makedirs(FLAGS.output_dataset_dir, exist_ok=True)

  with open(os.path.join(FLAGS.output_dataset_dir, "greek_epigraphy_dict.p"), "rb") as f:
    texts_dict = pickle.load(f)

  cnt_accents = Counter()
  cnt_noaccents = Counter()
  for t in texts_dict['train']:
    text = ' '.join([s + '.' for s in t.sentences])
    for word in re.findall(r'\w+', text):
      cnt_accents[word] += 1
      cnt_noaccents[strip_accents(word)] += 1

  with open(os.path.join(FLAGS.output_dataset_dir, "greek_epigraphy_wordlist.txt"), "w") as f:
    output = '\n'.join(['{}\t{}'.format(w, w_count) for w, w_count in cnt_accents.most_common()])
    f.write(output)

  with open(os.path.join(FLAGS.output_dataset_dir, "greek_epigraphy_wordlist.txt"), "w") as f:
    output = '\n'.join(['{}\t{}'.format(w, w_count) for w, w_count in cnt_noaccents.most_common()])
    f.write(output)


def main():
  # Process PHI to plain text format
  process_greek_epigraphy()

  # Generate most frequent words list
  generate_wordlist()


if __name__ == '__main__':
  main()
