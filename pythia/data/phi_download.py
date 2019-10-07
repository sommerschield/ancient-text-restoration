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
import concurrent.futures
import os

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

p = argparse.ArgumentParser(description='PHI download')
p.add_argument('--connections', default=100, type=int, metavar='N', help='number of connections')
p.add_argument('--timeout', default=5, type=int, metavar='N', help='seconds to timeout')
p.add_argument('--output', default=os.getcwd() + '/pythia/data/phi/', type=str, help='output path')
p.add_argument('--max_phi_id', default=400000, type=int, metavar='N', help='maximum phi inscription id')
FLAGS = p.parse_args()


# Auxiliary functions
def load_phi_id(phi_id, timeout, output):
  file_path = os.path.join(output, '{}.txt'.format(phi_id))
  if os.path.exists(file_path):
    return 'Exists'

  url_text_pattern = 'https://epigraphy.packhum.org/text/{}'
  url = url_text_pattern.format(phi_id)
  req = requests.get(url, timeout=timeout)

  if "Invalid PHI Inscription Number" not in req.text:
    try:
      soup = BeautifulSoup(req.text, "lxml")

      lines = []
      table = soup.find('table', attrs={'class': 'grk'})
      for row in table.find_all('tr'):
        tds = row.find_all('td')
        for td_i, td in enumerate(tds):
          if "class" in td.attrs and td.attrs["class"][0] == "id":
            continue
          lines.append(td.get_text().strip())

      text = "\n".join(lines)

      with open(file_path, 'w') as f:
        f.write(text)
    except:
      return 'Error'
  else:
    return 'Invalid'

  return 'Success'


def main():
  # Create structure
  os.makedirs(FLAGS.output, exist_ok=True)

  # Download inscriptions
  with concurrent.futures.ThreadPoolExecutor(max_workers=FLAGS.connections) as executor:
    future_to_phi = (executor.submit(load_phi_id, text_i, FLAGS.timeout, FLAGS.output) for text_i in
                     range(1, FLAGS.max_phi_id))
    for future in tqdm(concurrent.futures.as_completed(future_to_phi), total=FLAGS.max_phi_id):
      try:
        future.result()
      except:
        pass


if __name__ == '__main__':
  main()
