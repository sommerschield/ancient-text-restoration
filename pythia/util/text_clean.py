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
import unicodedata

from nltk import tokenize


def strip_accents(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def text_to_sentences(t, alphabet):
  # tokenize sentences and remove the empty ones
  sentences = []
  for s in tokenize.sent_tokenize(t):
    # remove all puntuation from sentence
    s = re.sub(r'[%s]+' % ''.join(alphabet.punctuation), ' ', s)

    # collapse spaces
    s = re.sub('\s+', ' ', s).strip()

    # append sentence
    if len(s) > 1:
      sentences.append(s)
  return sentences


def text_clean_phi(text_cleaned, alphabet):
  # Join lines ending with -
  text_cleaned = re.sub(r'-\n', r'', text_cleaned)

  # Remove Greek numerals with ʹ
  text_cleaned = re.sub(r'[\[\]\w]+ʹ', r'0', text_cleaned)

  # Remove Greek numerals
  word_boundary = '([\s\.\⏑\—\-\-\,⏑․]|$)'
  greek_numerals = re.escape('∶ΠT𐅈𐅃𐅉ϛ𐅀𐅁𐅂Ι𐅃ΔͰΗΧΜΤ𐅄𐅅𐅆𐅇𐅈𐅉𐅊𐅋𐅌𐅍𐅎𐅏𐅏𐅐𐅑��𐅓𐅔𐅕𐅖')
  text_cleaned = re.sub(r'\[[%s]+\]' % greek_numerals, r'', text_cleaned)
  text_cleaned = re.sub(
    r'%s[\[\]]?[%s]+[\[\]]?[%s]*%s' % (word_boundary, greek_numerals, greek_numerals, word_boundary),
    r'\1 0\2', text_cleaned)

  # Remove extra punctuation
  text_cleaned = re.sub(r'(\s*)[\∶|\⋮|\·|\⁙|\;]+(\s*)', r' ', text_cleaned)

  # Remove all (?)
  text_cleaned = re.sub(r'\s*\(\?\)', r'', text_cleaned)

  # Remove anything between {}
  text_cleaned = re.sub(r'{[^}]*}', r'', text_cleaned)

  # Remove parentheses surrounding greek characters
  text_cleaned = re.sub(r'\(([{}]+)\)'.format(''.join(alphabet.alphabet)), r'\1', text_cleaned)

  # Remove any parentheses that has content that is not within the greek alphabet
  text_cleaned = re.sub(r'\([^\)]*\)', r'', text_cleaned)

  # Remove vac, v v. vac vac. vac.? in (), etc
  text_cleaned = re.sub(r'(\d+\s)?\s*v[\w\.\?]*(\s\d+(\.\d+)?)?', '', text_cleaned)

  # Remove < >, but keep content
  text_cleaned = re.sub(r'<([^>]*)>', r'\1', text_cleaned)

  # Remove latin numbering within brackets [I]
  text_cleaned = re.sub(r'\[M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\]\s*$', r'', text_cleaned)

  # Lowercase
  text_cleaned = text_cleaned.lower()

  # filter alphabet to replace tonos and h
  text_cleaned = alphabet.filter(text_cleaned)

  # Convert short syllables to long ones
  text_cleaned = text_cleaned.replace('⏑', '—')
  text_cleaned = text_cleaned.replace('⏕', '—')
  text_cleaned = text_cleaned.replace('-', '—')

  # Collapse space between dashes (on purpuse double)
  text_cleaned = re.sub(r'—(?:[\s]+—)+', lambda g: re.sub(r'[\s]+', '', g.group(0)),
                        text_cleaned, flags=re.MULTILINE)

  # Replace dots with dashes and c.#
  text_cleaned = re.sub(r"(?:․|—)+\s?(?:c\.)?(\d+)(?:(\-|-)\d+)?\s?(․|—)+",
                        lambda g: alphabet.missing * int(g.group(1)),
                        text_cleaned, flags=re.MULTILINE)

  # Replace with missing character
  text_cleaned = text_cleaned.replace(u'\u2013', alphabet.missing)
  text_cleaned = text_cleaned.replace(u'\u2014', alphabet.missing)
  text_cleaned = text_cleaned.replace('․', alphabet.missing)

  # PHI #⁷removed automatically because only gr chars

  # Join ][ and []
  text_cleaned = text_cleaned.replace('][', '').replace('[]', '')

  # keep only alphabet characters
  chars = ''.join(alphabet.alphabet + alphabet.numerals + alphabet.punctuation + [
    alphabet.space, alphabet.missing, alphabet.sog, alphabet.eog])
  text_cleaned = re.sub(r'[^{}]'.format(re.escape(chars)), " ", text_cleaned)

  # remove space before punctuation / merge double punctuation
  chars = ''.join(alphabet.punctuation + [alphabet.eog])
  text_cleaned = re.sub(r'\s+([{}])'.format(re.escape(chars)), r'\1', text_cleaned)

  # remove leading space and punctuation
  text_cleaned = text_cleaned.lstrip(''.join(alphabet.punctuation + [alphabet.space]))

  # collapse spaces
  text_cleaned = re.sub('\s+', ' ', text_cleaned).strip()

  # collapse duplicate dots
  text_cleaned = re.sub(r'([{}])+'.format(''.join(alphabet.punctuation)), r'\1', text_cleaned)

  return text_cleaned
