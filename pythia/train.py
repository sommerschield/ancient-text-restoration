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
import importlib
import os
import pickle
import signal
from datetime import datetime

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

from .include.alphabet import GreekAlphabet
from .model.graph import build_graph
from .util import log
from .util.vocab import idx_to_text

#######################################
# Args
#######################################

p = argparse.ArgumentParser(description='Train')
# positional args:
p.add_argument('--exp_name', default='epigraphy', type=str, help='experiment name')
p.add_argument('--model', default='model_biword', type=str,
               help='model name, should be a file in model/, but excluding .py extension')
p.add_argument('--dataset', default=os.getcwd() + '/pythia/data/datasets/greek_epigraphy_dict.p', type=str,
               help='dataset file')
p.add_argument('--dataset_test_set', default='valid', type=str, help='valid/test set split')
p.add_argument('--load_checkpoint', default='', type=str, help='load from checkpoint')
# training behaviour options:
p.add_argument('--batch_size', default=32, type=int, help='batch size')
p.add_argument('--learning_rate', default=3e-4, type=float, help='learning rate')
p.add_argument('--grad_clip', default=5., type=float, help='gradient norm clipping')
p.add_argument('--beam_width', default=5, type=int, help='beam search width')
p.add_argument('--eval_samples', default=1600, type=int, help='number of evaluation samples')
p.add_argument('--summary_dir', default=os.getcwd() + '/tf/summary/', type=str, help='summary directory')
p.add_argument('--checkpoint_dir', default=os.getcwd() + '/tf/checkpoint/', type=str, help='checkpoint directory')
p.add_argument('--checkpoint_interval', default=500, type=int, help='checkpoint interval')
p.add_argument('--eval_interval', default=10000, type=int, help='eval interval')
p.add_argument('--report_interval', default=1000, type=int, help='report interval')
p.add_argument('--training_iterations', default=1000000, type=int, help='number of training iterations')
p.add_argument('--context_char_min', default=100, type=int, help='minimum context characters')
p.add_argument('--context_char_max', default=1000, type=int, help='minimum context characters')
p.add_argument('--pred_char_min', default=1, type=int, help='minimum pred characters')
p.add_argument('--pred_char_max', default=10, type=int, help='minimum pred characters')
p.add_argument('--missing_char_min', default=1, type=int, help='minimum missing characters')
p.add_argument('--missing_char_max', default=100, type=int, help='minimum missing characters')
p.add_argument('--pred_guess', default=False, type=bool, help='predict guessed characters')
p.add_argument('--log_dir', default=os.getcwd() + '/tf/log/', type=str, help='logging directory')
p.add_argument('--loglevel', default='INFO', type=str, metavar='LEVEL',
               help='Log level, will be overwritten by --debug. (DEBUG/INFO/WARN)')
FLAGS = p.parse_args()


def main():
  #######################################
  # Signal handler
  #######################################
  def handler(signum, frame):
    del signum, frame
    answer = input('Do you really want to exit? [yes]:')
    if answer == 'yes':
      print('bye')
      exit()
    print("Let's keep running")

  signal.signal(signal.SIGINT, handler)

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

  # summary writer
  cur_date = datetime.now().strftime("%Y_%m_%d %H:%M")
  os.makedirs(FLAGS.summary_dir, exist_ok=True)
  summary_writer_train = tf.summary.FileWriter(FLAGS.summary_dir + FLAGS.model + "_" + cur_date + '/train')
  summary_writer_valid = tf.summary.FileWriter(FLAGS.summary_dir + FLAGS.model + "_" + cur_date + '/valid')
  logging.info('Added summary writer and saver')

  #######################################
  # Train loop
  #######################################

  config = tf.ConfigProto()
  config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  checkpoint_dir = FLAGS.checkpoint_dir + '/' + FLAGS.model + "_" + FLAGS.exp_name + "_" + cur_date + '/'
  os.makedirs(checkpoint_dir, exist_ok=True)
  logging.info("Checkpoint dir: %s", checkpoint_dir)

  best_valid_cer = None
  train_loss_avg = None
  valid_cer_avg = None
  with tf.Session(config=config) as sess:
    sess.run(graph_tensors['init'])

    # Restore model weights from previously saved model
    if FLAGS.load_checkpoint:
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.load_checkpoint))
      logging.info("Model restored from file: %s" % FLAGS.load_checkpoint)

    start_iteration = sess.run(graph_tensors["global_step"])

    for train_iteration in range(start_iteration, FLAGS.training_iterations):
      if train_iteration > 0 and train_iteration % FLAGS.eval_interval == 0:

        out_tensors = sess.run({
          'train_loss': graph_tensors["train_loss"],
          'valid_cer': graph_tensors["valid_cer"],
          'valid_output': graph_tensors["valid_output"],
          'valid_batch': graph_tensors["valid_batch"],
          'summary_valid': graph_tensors["summary_valid"],
          'summary_train': graph_tensors["summary_train"],
          'train_step': graph_tensors["train_step"]})

        # average train loss
        if train_loss_avg is None:
          train_loss_avg = out_tensors['train_loss']
        else:
          train_loss_avg = train_loss_avg * 0.95 + out_tensors['train_loss'] * 0.05

        # average valid cer
        if valid_cer_avg is None:
          valid_cer_avg = out_tensors['valid_cer']
        else:
          valid_cer_avg = valid_cer_avg * 0.95 + out_tensors['valid_cer'] * 0.05

        # sample text
        x_ = idx_to_text(out_tensors['valid_batch']['x'][0, :out_tensors['valid_batch']['x_len'][0]], alphabet)
        y_ = idx_to_text(out_tensors['valid_batch']['y'][0, :out_tensors['valid_batch']['y_len'][0]], alphabet)
        y_pred_ = idx_to_text(
          out_tensors['valid_output'].sample[0, :, 0], alphabet)

        # print stats
        logging.info(
          "%d: train: %.8f, train_avg: %.8f, valid_cer: %.3f, valid_cer_avg: %.3f, y: %s, y_pred: %s, x: %s",
          train_iteration,
          out_tensors['train_loss'], train_loss_avg,
          out_tensors['valid_cer'], valid_cer_avg,
          y_, y_pred_, x_)

        # summaries
        summary_writer_train.add_summary(out_tensors['summary_train'], train_iteration)
        summary_writer_valid.add_summary(out_tensors['summary_valid'], train_iteration)

        if best_valid_cer is None or out_tensors['valid_cer'] < best_valid_cer:
          logging.info('Best model with cer %.3f. Saving to %s', out_tensors['valid_cer'],
                       checkpoint_dir)
          saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=train_iteration)
          best_valid_cer = out_tensors['valid_cer']

      else:
        out_tensors = sess.run(
          {'train_loss': graph_tensors["train_loss"],
           'summary_train': graph_tensors["summary_train"],
           'train_step': graph_tensors["train_step"]},
          options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

        # average train loss
        if train_loss_avg is None:
          train_loss_avg = out_tensors['train_loss']
        else:
          train_loss_avg = train_loss_avg * 0.95 + out_tensors['train_loss'] * 0.05

        # log statistics
        if train_iteration % FLAGS.report_interval == 0:
          logging.info("%d: train: %.8f, train_avg: %.8f",
                       train_iteration,
                       out_tensors['train_loss'], train_loss_avg)

        # summaries
        summary_writer_train.add_summary(out_tensors['summary_train'], train_iteration)


if __name__ == '__main__':
  main()
