#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_template.py: Unit Test for a module."""

import unittest
import math

import perceptron

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

class TestPerceptron(unittest.TestCase):
  def setUp(self):
    self.labels_weights = {"1":{"f1":0.5,
                                "f2":0.7,
                                "f3":0.2,
                                "next_bla":0.0,
                                "next_gmail":0.0},
                           "2":{"f1":0.9,
                                "f2":0.6,
                                "f3":0.6,
                                "next_bla":0.0,
                                "next_gmail":0.1},
                           "3":{"f1":0.4,
                                "f2":0.3,
                                "f3":0.8,
                                "next_bla":-0.1,
                                "next_gmail":0.0}}

  def tearDown(self):
    pass

  def test_classify(self):
    classifier = perceptron.Perceptron(self.labels_weights)
    fs1 = {"f1": 1.0, "f2": 9.0, "f3":1.0}
    fs2 = {"f1": 8.0, "f2": 1.0, "f3":1.0}
    fs3 = {"f1": 1.0, "f2": 1.0, "f3":8.0}
    self.assertEqual("1", classifier.classify(fs1))
    self.assertEqual("2", classifier.classify(fs2))
    self.assertEqual("3", classifier.classify(fs3))

  def almost_equal(self, num1, num2):
    return 0.0 == round(math.fabs(num1-num2), 3)

  def test_update(self):
    classifier = perceptron.Perceptron(self.labels_weights)
    classifier.learning_rate = 0.1
    classifier.update({"f1": 1.0, "f2": 9.0, "f3":1.0, "next": "bla"}, "1")

    label1 = {"f1":0.6, "f2":1.6, "f3":0.3, "next_bla": 0.1, "next_gmail":0.0}
    for key in classifier.labels_weights["1"]:
      self.assertTrue(self.almost_equal(label1[key],
                      classifier.labels_weights["1"][key]))

    label2 = {"f1":0.8, "f2":-0.3, "f3":0.5, "next_gmail":0.1, "next_bla":-0.1}
    for key in classifier.labels_weights["2"]:
      self.assertTrue(self.almost_equal(label2[key],
                      classifier.labels_weights["2"][key]))

    
if __name__ == "__main__":
  unittest.main()
