#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_template.py: Unit Test for a module."""

import unittest
import json
import parse_talk

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

class TestClass(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testBolshoi(self):
    answers = json.loads(open('bolshoi.ans.json', 'r').read())
    text = open('bolshoi.txt', 'r').read()
    text = text.decode('utf-8')
    self.assertEqual(answers, parse_talk.parse_page(text))

if __name__ == "__main__":
  unittest.main()   

