#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
from numpy import *
from cPickle import dump, load
import json
import pdb
import os

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

CACHE = True


class Serialized(object):
  """Decorator that serliazes a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned, and
  not re-evaluated.
  """

  __CACHE = 'cache'
  if not os.path.isdir(__CACHE):
    os.mkdir(__CACHE)

  def __init__(self, func):
    self.func = func

  def _path(self, name):
    filename = '.'.join([self.func.__module__, self.func.__name__, name, 'ser'])
    return os.path.join(Serialized.__CACHE, filename)

  def __call__(self, name, *args):
    data = None
    filename = self._path(name)
    try:
      if not CACHE:
        raise Exception
      fh = open(filename, 'r')
      data = load(fh)
      fh.close()
    except:
      data = self.func(*args)
      if CACHE:
        fh = open(filename, 'w')
        dump(data, fh)
        fh.close()
    return data

  def __get__(self, obj, objtype):
    """Support instance methods."""
    return functools.partial(self.__call__, obj)


class Memoized(object):
  """Decorator that caches a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned, and
  not re-evaluated.
  """

  __cache = {}

  def __init__(self, func):
    self.func = func
    self.key = (func.__module__, func.__name__)

  def __call__(self, *args):
    try:
      return Memoized.__cache[self.key][args]
    except KeyError:
      value = self.func(*args)
      if self.key in Memoized.__cache:
        Memoized.__cache[self.key][args] = value
      else:
        Memoized.__cache[self.key] = {args : value}
      return value
    except TypeError:
      # uncachable -- for instance, passing a list as an argument.
      # Better to not cache than to blow up entirely.
      return self.func(*args)

  def __get__(self, obj, objtype):
    """Support instance methods."""
    return functools.partial(self.__call__, obj)

  @staticmethod
  def reset():
    Memoized.__cache = {}


def indices(list_, item):
  inds = []
  for i in range(len(list_)):
    if list_[i] == item:
      inds.append(i)
  return inds


def z_score(list_):
  m = float(mean(array(list_)))
  s = float(std(array(list_)))
  return [(x-m)/s for x in list_]


def anomalies(list_, threshold):
  z_scores = z_score(list_)
  anomaly = []
  for x in z_scores:
    if x>threshold:
      anomaly.append(1)
    else:
      anomaly.append(0)
  return zip(list_, anomaly)


def featureset(distributions, elements, size):
  featureset = set()
  for lang in distributions.keys():
    featureset.update(distributions[lang][elements][size].keys())
  return featureset


def freqdist_prob(freqdist, item):
  count = freqdist.get(item, 0.0)
  return count/float(freqdist.N())
    
def valid_feature(distributions, elements, size, item):
  empty = 0
  languages = sorted(distributions.keys())
  for lang in languages:
    count = distributions[lang][elements][size].get(item, 0.0)
    if count < 2:
      empty += 1
  if empty > len(languages)/2:
    return False
  return True

def features_probs(distributions, elements, size):
  features = featureset(distributions, elements, size)
  languages = sorted(distributions.keys())
  features_distribution = {}
  for feature in features:
    if valid_feature(distributions, elements, size, feature):
      feature_probs = []
      for lang in languages:
        prob = freqdist_prob(distributions[lang][elements][size], feature)
        feature_probs.append(prob)
      features_distribution[feature] = feature_probs
  return features_distribution

@Serialized
def features_languages(distributions):
  languages = sorted(distributions.keys())
  elements = distributions[languages[0]].keys()
  sizes = distributions[languages[0]][elements[0]].keys()
  probabilities = {}
  for element in elements:
    for size in sizes:
      probabilities[(element,size)] = features_probs(distributions, element, size)
  return probabilities


def interesting(probabilities, threshold):
  results = []
  for element_size in probabilities:
    features = probabilities[element_size]
    i = 0
    j = 0
    for feature in features:
      abnormal = anomalies(features[feature], threshold)
      probs, flags = zip(*abnormal)
      flags_indices = indices(flags, 1)
      if len(flags_indices) > 0:
        values = [x for (x,y) in abnormal if y==1]
        result = zip(flags_indices, values)
        i += 1
        results.append((feature,result))
      else:
        j += 1
    logging.info("%d features found in %s: %d interesting", i+j, str(element_size), i)
  return results


def main(options, args):
  distributions = load(open(options.filename, 'r'))
  logging.info("File has been read")
  languages = sorted(distributions.keys())
  map_ = dict(zip(range(len(languages)), languages))
  probabilties = features_languages('features_probs', distributions)
  logging.info("Probabilities are caclulated for each feature")
#  pdb.set_trace()
  facts = interesting(probabilties, 2.235)
  logging.info("facts are caclulated for each feature")
  renamed_facts = []
  for fact in facts:
    feature, probs = fact
    renamed = {}
    for index, value in probs:
      renamed[map_[index]] = value
    renamed_facts.append((feature, renamed))
  json.dump(renamed_facts, open(options.filename+'.facts', 'w'))
  logging.info("facts are dumped")
 

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

