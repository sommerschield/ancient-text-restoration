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
import collections
import importlib
import os
import pickle

import editdistance
import tensorflow as tf

from .include.alphabet import GreekAlphabet
from .model.graph import build_graph
from .util import log
from .util.text_clean import strip_accents
from .util.vocab import idx_to_text

#######################################
# Args
#######################################

p = argparse.ArgumentParser(description='Test')
# positional args:
p.add_argument('--model', default='model_biword', type=str, help=('experiment name, should be a file in model/, '
                                                                  'but excluding .py extension'))
p.add_argument('--dataset', default=os.getcwd() + '/pythia/data/datasets/greek_epigraphy_dict.p', type=str,
               help='dataset file')
p.add_argument('--dataset_test_set', default='valid', type=str, help='valid/test set split')
p.add_argument('--load_checkpoint', default=os.getcwd() + '/tf/checkpoint/your_model/', type=str,
               help='load from checkpoint')
p.add_argument('--use_accents', default=False, type=bool, help='use accents')
# training behaviour options:
p.add_argument('--batch_size', default=64, type=int, help='batch size')
p.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
p.add_argument('--grad_clip', default=5., type=float, help='gradient norm clipping')
p.add_argument('--beam_width', default=20, type=int, help='beam search width')
p.add_argument('--log_samples', default=False, type=bool, help='log samples')
p.add_argument('--eval_samples', default=64, type=int, help='number of evaluation samples')
p.add_argument('--test_iterations', default=200, type=int, help='number of training iterations')
p.add_argument('--context_char_min', default=-1, type=int, help='minimum context characters')
p.add_argument('--context_char_max', default=1000, type=int, help='minimum context characters')
p.add_argument('--pred_char_min', default=1, type=int, help='minimum pred characters')
p.add_argument('--pred_char_max', default=10, type=int, help='minimum pred characters')
p.add_argument('--missing_char_min', default=0, type=int, help='minimum missing characters')
p.add_argument('--missing_char_max', default=0, type=int, help='minimum missing characters')
p.add_argument('--pred_guess', default=False, type=bool, help='predict guessed characters')
p.add_argument('--log_dir', default=os.getcwd() + '/tf/log/', type=str, help='logging directory')
p.add_argument('--results_dir', default=os.getcwd() + '/tf/results/', type=str, help='results directory')
p.add_argument('--loglevel', default='INFO', type=str, metavar='LEVEL',
               help='Log level, will be overwritten by --debug. (DEBUG/INFO/WARN)')
FLAGS = p.parse_args()


def main():
  #######################################
  # Logging
  #######################################

  log.init(FLAGS)
  logging = log.get(__file__, FLAGS)
  for f in vars(FLAGS):
    logging.info('--%s=%s' % (f, getattr(FLAGS, f)))

  #######################################
  # Alphabet
  #######################################

  alphabet = GreekAlphabet()
  logging.info('Loaded alphabet')

  #######################################
  # Dataset
  #######################################

  with open(FLAGS.dataset, "rb") as f:
    texts = pickle.load(f)
  logging.info('Loaded data')

  #######################################
  # Graph
  #######################################

  tf.reset_default_graph()

  # load model
  model_module = importlib.import_module('pythia.model.' + FLAGS.model)
  model = model_module.Model(config=FLAGS, alphabet=alphabet)
  logging.info('Loaded model')

  # build graph
  graph_tensors = build_graph(FLAGS, alphabet, texts, model)
  logging.info('Built graph')

  # configure a checkpoint saver.
  saver = tf.train.Saver()

  #######################################
  # Test loop
  #######################################

  with tf.Session() as sess:
    sess.run(graph_tensors['init'])

    # Restore model weights from previously saved model
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.load_checkpoint))
    logging.info("Model restored from file: %s" % FLAGS.load_checkpoint)

    correct_preds = []
    cer_target = ''
    cer_pred = ''
    for test_iteration in range(FLAGS.test_iterations):

      out_tensors = sess.run({
        'valid_output': graph_tensors["valid_output"],
        'valid_batch': graph_tensors["valid_batch"]})

      # sample text
      for b in range(FLAGS.batch_size):
        x_len_ = out_tensors['valid_batch']['x_len'][b]
        y_len_ = out_tensors['valid_batch']['y_len'][b]
        x_ = idx_to_text(out_tensors['valid_batch']['x'][b, :x_len_], alphabet)
        y_ = idx_to_text(out_tensors['valid_batch']['y'][b, :y_len_], alphabet)

        if FLAGS.log_samples:
          logging.info('******** %d/%d ********', test_iteration, b)
          logging.info("x: %s", x_)
          logging.info("y: %s", y_)

        # Compute CER
        y_pred_0 = idx_to_text(
          out_tensors['valid_output'].sample[b, :, 0], alphabet)
        cer_target += y_
        cer_pred += y_pred_0

        # Mark correct prediction
        correct_pred = 0
        for b_i in range(out_tensors['valid_output'].sample.shape[2]):
          y_pred_ = idx_to_text(
            out_tensors['valid_output'].sample[b, :, b_i], alphabet)
          if FLAGS.log_samples:
            logging.info(
              "y_%d: %s", b_i, y_pred_)
          if strip_accents(y_) == strip_accents(y_pred_):
            correct_pred = b_i + 1

        correct_preds.append(correct_pred)

    if FLAGS.use_accents:
      cer = editdistance.eval(y_, y_pred_0) / float(len(y_))
    else:
      cer = editdistance.eval(strip_accents(cer_target), strip_accents(cer_pred)) / float(len(cer_target))

    # Print counts
    logging.info('******** Performance ********')
    correct_preds_count = collections.Counter(correct_preds)
    logging.info('cer: {}'.format(cer))
    for k in sorted(correct_preds_count.keys()):
      logging.info('top-{}: {}'.format(k, float(correct_preds_count[k]) / len(correct_preds)))

    # Write results
    model_path = os.path.basename(os.path.normpath(FLAGS.load_checkpoint))
    results_path = os.path.join(
      FLAGS.results_dir, model_path,
      '{dataset}_{samples}_{context_min}-{context_max}_{pred_min}-{pred_max}_{guess}_{accents}.txt'.format(
        samples=FLAGS.test_iterations * FLAGS.batch_size,
        context_min=FLAGS.context_char_min,
        context_max=FLAGS.context_char_max,
        pred_min=FLAGS.pred_char_min,
        pred_max=FLAGS.pred_char_max,
        dataset=os.path.splitext(os.path.basename(FLAGS.dataset))[0],
        guess='predg' if FLAGS.pred_guess else 'pred',
        accents='acc' if FLAGS.use_accents else 'noacc',
      ))

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, 'w', encoding='utf8') as f:
      f.write('model,{}\n'.format(model_path))
      f.write('cer,{}\n'.format(cer))
      for k in sorted(correct_preds_count.keys()):
        f.write('top-{},{}\n'.format(k, float(correct_preds_count[k]) / len(correct_preds)))


if __name__ == '__main__':
  main()
