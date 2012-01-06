#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Collection of useful classes and methods that are useful."""

from optparse import OptionParser
import logging
from multiprocessing import Pool, cpu_count
import functools
import os
from cPickle import dump, load

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
