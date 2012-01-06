#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
import cPickle
from math import log
from itertools import combinations
import json
from pylab import *
import util
from util import *

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
CACHE = True

def kl(freqDist1, freqDist2):
  samples = freqDist1.samples() + freqDist2.samples()
  total = 0.0
  p_n = float(freqDist1.N())
  q_n = float(freqDist2.N())
  for sample in samples:
    p = freqDist1.get(sample, 0.0)/p_n
    q = freqDist2.get(sample, 0.0)/q_n
    if p*q == 0:
      total += 0
    else:
      total += p*log(p/q)
  return total


def kl_languages(lang1, lang2):
  total = 0.0
  for feature in lang1:
    for size in lang1[feature]:
      total += kl(lang1[feature][size], lang2[feature][size])
  return total


@Serialized
def distributions_KLs(filename):
  distributions = cPickle.load(open(filename, 'r'))
  logging.info("%s has been read", options.filename)
  languages = distributions.keys()
  KLs = {}
  for lang in languages:
    KLs[lang] = {}

  for lang1,lang2 in combinations(languages, 2):
    KLs[lang1][lang2] = kl_languages(distributions[lang1], distributions[lang2])
    KLs[lang2][lang1] = kl_languages(distributions[lang2], distributions[lang1])
    KLs[lang1][lang1] = 0.0
    KLs[lang2][lang2] = 0.0
  return KLs


def main(options, args):
  util.CACHE = CACHE
  util.CACHE_FOLDER = os.path.dirname(os.path.abspath(options.filename))
  logging.info("Script started to calculate KL divergence")
  KLs = distributions_KLs('dists_kls', options.filename)
  languages = KLs.keys()
  logging.info("KL divergence has been calculated")
  cfm = []
  for lang in languages:
    cfm.append([int(5.0*(KLs[lang][l] + KLs[l][lang]))/-10.0 for l in languages])

  new_languages = cluster_table(cfm, languages)
  if len(new_languages) != len(languages):
    logging.fatal("Missing languages!")

  logging.info("Old order of languages %s", str(languages))
  logging.info("New order of languages %s", str(new_languages))
  
  cfm2 = []
  for lang in new_languages:
    cfm2.append([int(5.0*(KLs[lang][l]+KLs[l][lang]))/10.0 for l in new_languages])
  
  json.dump((KLs, cfm2, new_languages), open(options.filename+'.kl.log', 'w'), indent=2)
  cfm_map(cfm2, new_languages, 'KL divergence', filename=options.filename+'.kl.png')
 

if  __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)
