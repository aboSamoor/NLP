#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Wikipedia.py: Interface with wikipedia database."""

from optparse import OptionParser
import logging
import MySQLdb as mdb
import sys
from collections import namedtuple, OrderedDict
import json
import re
from util import pool_map 
from parse_talk import parse_page
from nltk.tag.senna import POSTagger


__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

USER = "rmyeid"
#DB = "wikipedia"
DB = "current"
PASSWORD = "bla"
HOST = "carnap"

LANGUAGE_CATEGORY = re.compile('^User_[^_]*?-[^_]*?$')

Revision = namedtuple("Revision", ("rev_id rev_page rev_text_id rev_comment "
                                    "rev_user rev_user_text rev_timestamp "
                                    "rev_minor_edit rev_deleted rev_len rev_parent_id"))

Text = namedtuple("Text", ("old_id old_text old_flags"))

Category = namedtuple("Category", ("cl_from cl_to cl_sortkey cl_timestamp cl_sortkey_prefix cl_collation cl_type"))

Comment = namedtuple("Comment", ("autoid user_name page_id page_title comment time_stamp"))

TalkPage = namedtuple("TalkPage", ("old_id page_id old_text page_title"))


class WikipediaDB(object):
  
  def __init__(self, host=HOST, user=USER, password=PASSWORD, db=DB):
    self.conn = mdb.connect(host, user, password, db);

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

  def build_user_language_dictionary(self):
    query = "select * from language where cl_to like 'User\_%-%';"
    users = {}
    for cat in self._execute(query):
      category = Category(*cat)
      username = category.cl_sortkey.replace('\n','/').split('/', 1)[0].replace(' ', '_')
      language = category.cl_to
      if LANGUAGE_CATEGORY.match(language):
        if username in users:
          users[username].append(language)
        else:
          users[username] = [language]
    for user in users:
      users[user] = list(set(users[user]))
    logging.debug('%d users were found' % len(users))
    return users

  def text(self, text_id):
    query = "select * from text where old_id=%d;" % text_id
    return Text(*self._execute(query).next())

  def _execute(self, query):
    cursor = self.conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    while row is not None:
      yield row
      row = cursor.fetchone()
    cursor.close()

  def version(self):
    query = "SELECT VERSION()"
    return self._execute(query)

  @staticmethod
  def insert_statement(ordered_dict, table):
    structure = '(%s)' % ','.join(ordered_dict.keys())
    values = '(%s)' % ', '.join('%'+'(%s)s' % key  for key in ordered_dict.keys())
    statement = 'insert into %s %s values %s' % (table, structure, values)
    return statement

  def insert(self, ordered_dict, table):
    cursor = self.conn.cursor()
    statement = WikipediaDB.insert_statement(ordered_dict, table)
    cursor.execute(statement, dict(ordered_dict))
    cursor.close()

  def talk_pages(self, table):
    query = "select * from %s;" % table
    for page in self._execute(query):
      yield TalkPage(*page)

  def get_comment_by_autoid(self, autoid):
    query = "select * from comments where autoid=%s;" % autoid
    pages = [page for page in self._execute(query)]
    if len(pages) > 1:
      logging.debug("we recieved %d pages for id=%s", len(pages), autoid)
    return Comment(*pages[0])

users_langs = None

def get_details(comment_details):
  category_level = {'User_en-5', 'User_en-4', 'User_en-3', 'User_en-2', 'User_en-1', 'User_en-0'}
  comment, row = comment_details
  if not row:
    return tuple()
  autoid, text, lang = comment
  if lang == 'User_en-us':
    level = 'User_en-N'
  else:
    langs = set(users_langs[row.user_name])
    common_levels = langs.intersection(category_level)
    level = list(common_levels)[0] if common_levels else "UNK"
    if level == 'User_en-N':
      level = "UNK"
  return (autoid, text, lang, row.user_name, row.page_id, row.page_title,
                     row.time_stamp, level)


def add_columns_to_data():
  global users_langs
  logging.info("Started running")
  data = json.load(open(options.filename, 'rb'))
  users_langs = json.load(open('users_languages_dictionary.json', 'rb'))
  db = WikipediaDB()
  rows = [db.get_comment_by_autoid(str(comment[0])) for comment in data]
  db.close()
  logging.info("Fetched all the data needed from the database")
  combined = zip(data, rows)
  new_data = filter(lambda x: x, [get_details(e) for e in combined])
  logging.info("Number of original comments is %d", len(data))
  logging.info("Number of new comments is %d", len(new_data))
  json.dump(new_data, open(options.filename+'.details', 'wb'), indent=2)


def main(options, args):
  total_comments = 0
  total_pages = 0
  total_users_known_comments = 0
  last_page_id = 0
  fh = open(options.filename, 'r')
  users = json.load(fh)
  logging.info("Users database is loaded")
  try:
    db = WikipediaDB()
    logging.info("Connected to database")
    for page in db.talk_pages('talk_pages_1'):
      comments = parse_page(page.old_text)
      total_pages += 1
      total_comments += len(comments)
      last_page_id = page.page_id
      found = 0
      for comment in comments:
        username = comment[1]["username"].upper()
        if username in users:
          row = OrderedDict()
          row["user_name"] = username
          row["page_id"] = page.page_id
          row["page_title"] = page.page_title
          row["comment"] = comment[0]
          row["time_stamp"] = comment[1].get("time", "0")
          db.insert(row, 'comments')
          found += 1
      total_users_known_comments += found
      logging.debug('Found %d known users comments out of %d comments in page %s',
                    found, len(comments), page.page_id)
      if total_pages % 10000 == 0:
        logging.info('\nParsed %d pages. Latest page is %d. Known comments = %d '
                     'found = %d. Comments/page = %.2f. Known comments = %.2f.',
                     total_pages,
                     last_page_id,
                     total_users_known_comments, total_comments,
                     total_comments/float(total_pages),
                     total_users_known_comments/float(total_comments))
        
    logging.info('Parsed %d pages, latest page is %d. Known users comments are %d out of %d', total_pages, last_page_id, total_users_known_comments, total_comments)
    db.close()
  except mdb.Error, e:
    print "Error %d: %s" % (e.args[0], e.args[1])
    sys.exit(1)



if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  add_columns_to_data()
