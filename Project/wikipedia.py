#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Wikipedia.py: Interface with wikipedia database."""

from optparse import OptionParser
import logging
import MySQLdb as mdb
import sys
from collections import namedtuple

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

USER = "rmyeid"
DB = "wikipedia"
PASSWORD = "bla"
HOST = "localhost"

Revision = namedtuple("Revision", ("rev_id rev_page rev_text_id rev_comment "
                                    "rev_user rev_user_text rev_timestamp "
                                    "rev_minor_edit rev_deleted rev_len rev_parent_id"))

Text = namedtuple("Text", ("old_id old_text old_flags"))


class WikipediaDB(object):
  
  def __init__(self, host=HOST, user=USER, password=PASSWORD, db=DB):
    self.conn = mdb.connect(HOST, USER, PASSWORD, DB);

  def close(self):
    self.conn.close()

  def revisions(self, page):
    query = "select * from revision where rev_page=%d order by rev_id;" % page
    rev_previous = None
    for revision in self._execute(query):
      revision = Revision(*revision)
      if rev_previous:
        revision = revision._replace(rev_parent_id=rev_previous.rev_id)
        rev_previous = revision
      else:
        revision = revision._replace(rev_parent_id= -1)
        rev_previous = revision
      yield revision

  def text(self, text_id):
    query = "select * from text where old_id=%d;" % text_id
    return Text(*self._execute(query).next())

  def _execute(self, query):
    cursor = self.conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    while row:
      yield row
      row = cursor.fetchone()
    cursor.close()

  def version(self):
    query = "SELECT VERSION()"
    return self._execute(query)

def main(options, args):
  try:
    db = WikipediaDB()
    data = db.revisions(12).next()
    db.close()
  except mdb.Error, e:
    print "Error %d: %s" % (e.args[0],e.args[1])
    sys.exit(1)

  print data


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level)
  main(options, args)

