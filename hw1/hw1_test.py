#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_template.py: Unit Test for a module."""

import unittest
import os

import hw1

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

TEST_DIR = os.path.abspath('testdata/')
TEST_FILE0 = os.path.join(TEST_DIR, 'test0.txt')
text = ""

def setUpModule():
  global text
  text = open(TEST_FILE0, 'r').read()

class TestClass(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_examples_set(self):
    training_set = hw1.get_examples_set(text)
    self.assertEqual(9, len(training_set))

  def testGetFeatureSet(self):
    statement = ("The girls %s the school played on ground ." 
                  % hw1.BLANK_POSITION)
    features1 = hw1.get_featureset(statement.split(hw1.SPACE))
    expected_features1 = {"next": "the", "previous": "girls"}
    self.assertEqual(features1, expected_features1)

    features2 = hw1.get_featureset([hw1.BLANK_POSITION])
    expected_features2 = {"next": hw1.STATEMENT_END,
                          "previous": hw1.STATEMENT_START}
    self.assertEqual(features2, expected_features2)

if __name__ == "__main__":
  unittest.main() 
