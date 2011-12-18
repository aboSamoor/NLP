#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""split.py: Slice a list a dump it as a json file."""

from optparse import OptionParser
import logging
import json

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def main(options, args):
  data = json.load(open(options.filename, 'r'))
  p = int(options.percentage)/100.0
  limit = int(len(data) * p)
  json.dump(data[:limit], open(options.filename+"."+options.percentage, "w"))

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-p", "--percentage", dest="percentage", help="10")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

