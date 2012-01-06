#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Collection of useful classes and methods that are useful."""

from optparse import OptionParser
import logging
from multiprocessing import Pool, cpu_count
import functools
import os
from copy import deepcopy
from cPickle import dump, load
from sklearn import cluster
import numpy as np
from pylab import figure, title, xticks, yticks, savefig, plt

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

CACHE_FOLDER = 'cache'
CACHE = True

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


class Serialized(object):
  """Decorator that serliazes a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned, and
  not re-evaluated.
  """

  def __init__(self, func):
    self.func = func

  def _path(self, name):
    filename = '.'.join([self.func.__module__, self.func.__name__, name, 'ser'])
    return os.path.join(CACHE_FOLDER, filename)

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


def pool_map(func, list_, num_processors=2):
  num_processors  = cpu_count()/3
  p = Pool(num_processors)
  results = p.map(func, list_)
  p.close()
  p.join()
  return results


def cluster_table(table, labels):
  c = cluster.AffinityPropagation().fit(table)
  groups_centers = list(c.cluster_centers_indices_)
  new_labels = []
  centers_sizes = []

  groups = []
  for center in groups_centers:
    mark = c.labels_[center]
    size = len(filter(lambda x: x==mark, c.labels_))
    group = []
    for i in range(len(c.labels_)):
      if c.labels_[i] == mark and i!=center:
        group.append(labels[i])
    group.insert(len(group)/2, labels[center])
    groups.append(group)
    logging.info("Following labels are grouped %s. %s is the center", str(group), labels[center])

  ordered_groups = [y for x,y in  sorted([(len(x),x) for x in groups])]
  new_labels = []
  for group in ordered_groups:
    new_labels.extend(group)
  return new_labels


def unlabel_table(labeled_table):
  labels = sorted(labeled_table.keys())
  table = []
  for label in labels:
    table.append([labeled_table[label][l] for l in labels])
  return table,labels  


def reorder_table(labeled_table, labels):
  table = []
  for label in labels:
    table.append([labeled_table[label][l] for l in labels])
  return table


def label_table(unlabeled_table, labels):
  labeled_table = dict(zip(labels, [{} for l in labels]))
  map_ = dict(zip(range(len(labels)), labels))
  width = len(unlabeled_table)
  height = len(unlabeled_table[0])
  for x in range(width):
    for y in range(height):
      labeled_table[map_[x]][map_[y]] = unlabeled_table[x][y]
  return labeled_table


def clustered_cfm(cfm, labels, title_='', filename=''):
  new = deepcopy(cfm)
  for i in range(len(new)):
    for j in range(len(new[0])):
      new[i][j] = (cfm[i][j]+cfm[j][i])/2.0
      
  labeled_CFM = label_table(new, labels)
  LANGS_order = cluster_table(new, labels)
  new_reordered = reorder_table(labeled_CFM, LANGS_order)
  return cfm_map(new_reordered, LANGS_order, title_, filename)


def cfm_map(cfm, labels, title_='', filename=''):
  fig = figure()
  title(title_)
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(cfm), cmap=plt.cm.jet, interpolation='nearest')
  width = len(cfm)
  height = len(cfm)
  for x in xrange(width):
     for y in xrange(height):
       ax.annotate(str(cfm[x][y]), xy=(y, x),
                     horizontalalignment='center',
                     verticalalignment='center')
  cb = fig.colorbar(res)
  xticks(range(width), labels)
  yticks(range(height), labels)
  savefig(filename, format='png')
  return fig


def main(options, args):
  pass

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)
