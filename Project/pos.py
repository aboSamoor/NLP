#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
import json
from multiprocessing import Pool, cpu_count, Lock
from nltk.tag import stanford
import senna

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

#tagger = stanford.StanfordTagger('/media/data/NER/stanford/pos/models/left3words-wsj-0-18.tagger',
#                                 '/media/data/NER/stanford/pos/stanford-postagger.jar',
#                                 encoding='utf-8')
tagger = senna.SennaTagger('/media/petra/NER/senna-v2.0', encoding='utf-8')
i = 0
size = 0
samples = []
lock = Lock()

def process(labeled_comments):
  global i
  comments, labels = zip(*labeled_comments)
  tagged_comments = tagger.corpus_tag(comments)
  results = zip(tagged_comments, labels)
  lock.acquire()
  i += len(labels)
  lock.release()
  percentage = '%.6f' % (i/float(len(samples)))
  print percentage
  return results

def clean_samples(labeled_samples):
  comments, labels = zip(*labeled_samples)
  clean_comments = []
  for comment in comments:
    clean_comments.append(filter(lambda x:x, comment))
  clean_comments = filter(lambda x: x and len(x[0]) > 1, clean_comments)
  results = zip(clean_comments, labels)
  return results

def main(options, args):
  global samples
  samples = json.load(open(options.filename, 'r'))
  p = Pool(cpu_count()-1)
  size = len(samples)/(25*cpu_count())
  splitted_samples = [clean_samples(samples[i:i+size]) for i in range(0, len(samples), size)]
#  for sample in new_samples:
#    process(sample)
  results = p.map(process, splitted_samples)
  json.dump(results, open(options.filename+'.pos', 'w'))


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

