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

import datetime
import logging
import os

import coloredlogs

# logging handlers
handler_file = None


def init(args):
  global handler_file
  logfile = os.path.join(args.log_dir, args.model,
                         'log_' + datetime.datetime.now().isoformat('T').split('.')[0] + '.txt')
  os.makedirs(os.path.join(args.log_dir, args.model), exist_ok=True)
  handler_file = logging.FileHandler(logfile)
  handler_file.setLevel(getattr(logging, args.loglevel))
  handler_file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler_file.setFormatter(handler_file_fmt)


def get(filename, args):
  logger = logging.getLogger(os.path.basename(filename))
  logger.setLevel(getattr(logging, args.loglevel))
  coloredlogs.install(level=args.loglevel)
  logger.addHandler(handler_file)
  return logger
