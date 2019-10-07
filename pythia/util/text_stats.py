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


def texts_statistics(texts, frequencies=True, missing_symbol='-'):
  count = {'sentences': 0, 'words': 0, 'texts': len(texts)}
  words = {}
  char = {}

  for t in texts:
    count['sentences'] += len(t.sentences)
    for s in t.sentences:
      tok = s.split()
      count['words'] += len(tok)

      # count frequencies
      if frequencies:
        # word counts
        for w in tok:
          if w in words:
            words[w] += 1
          else:
            words[w] = 1
        # character counts
        for c in s:
          if c in char:
            char[c] += 1
          else:
            char[c] = 1

  print(count)

  if frequencies:
    print('Word frequencies')
    for k, v in sorted(words.items(), key=lambda x: x[1], reverse=True)[:20]:
      print('-- %s: %s' % (k, v))

    print('Character frequencies')
    for k, v in sorted(char.items(), key=lambda x: x[1], reverse=True):
      print('-- %s: %s' % (k, v))

    if missing_symbol in char:
      print('Missing frequencies')
      print(float(char[missing_symbol]) / (sum(char.values()) - char[missing_symbol]))
