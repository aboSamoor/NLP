# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the Senna tagger
#
# Copyright (C) 2001-2011 NLTK Project
# Author: Rami Al-Rfou' <rmyeid@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
#
# $Id: senna.py $

"""
A module for interfacing with the SENNA tagger.
"""

import os
import subprocess
import tempfile
import nltk
from nltk.tag.api import *

_senna_url = 'http://ml.nec-labs.com/senna/'

class SennaTagger(TaggerI):
    """
    A class for pos tagging with Senna Tagger. The input is the paths to:
     - A path to the  

    Example:

        >>> tagger = senna.SennaTagger(path='/media/data/NER/senna-v2.0')
        >>> tagger.tag('What is the airspeed of an unladen swallow ?'.split())
        [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'),
        ('of', 'IN'), ('an', 'DT'), ('unladen', 'JJ'), ('swallow', 'VB'), ('?', '.')]
    """
    def __init__(self, path, encoding=None, verbose=False):
        self._encoding = encoding
        self._path = os.path.normpath(path) + os.sep

    def tag(self, tokens):
        return self.batch_tag([tokens])[0]

    def batch_tag(self, sentences):
        encoding = self._encoding

        executable_ = os.path.join(self._path, 'senna-linux64')
        # Build the senna command to run the tagger
        _senna_cmd = [executable_, '-path', self._path, '-usrtokens', '-pos']

        # Write the actual sentences to the temporary input file
        _input = '\n'.join((' '.join(x) for x in sentences))
        if isinstance(_input, unicode) and encoding:
            _input = _input.encode(encoding, 'replace')

        # Run the tagger and get the output
        p = subprocess.Popen(_senna_cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate(input=_input)
        senna_output = stdout

        # Check the return code.
        if p.returncode != 0:
            print stderr
            raise OSError('Senna command failed!')

        if encoding:
            senna_output = stdout.decode(encoding, 'replace')

        # Output the tagged sentences
        tagged_sentences = [[]]
        for tagged_word in senna_output.strip().split("\n"):
            if not tagged_word:
                tagged_sentences.append([])
                continue
            word, tag = tagged_word.split('\t')
            tagged_sentences[-1].append((word.strip(), tag.strip()))
        return tagged_sentences

    def _process_line(self, line):
        words = line.strip().split('\t')
        return (words[0].strip(), words[1].strip())
    
    def _process_statement(self, statement):
        lines = statement.strip().split('\n')
        return [self._process_line(line) for line in lines]

    def _process_article(self, article):
        return [self._process_statement(statement) for statement in article]

    def corpus_tag(self, articles):
        encoding = self._encoding

        executable_ = os.path.join(self._path, 'senna-linux64')
        # Build the senna command to run the tagger
        _senna_cmd = [executable_, '-path', self._path, '-usrtokens', '-pos']

        # Write the actual sentences to the temporary input file
        lengths = [len(article) for article in articles]
        _input ='\n'.join(['\n'.join([' '.join(sentence) for sentence in article]) for article in articles])
        if isinstance(_input, unicode) and encoding:
            _input = _input.encode(encoding, 'replace')

        # Run the tagger and get the output
        p = subprocess.Popen(_senna_cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate(input=_input)
        senna_output = stdout

        # Check the return code.
        if p.returncode != 0:
            print stderr
            raise OSError('Senna command failed!')

        if encoding:
            senna_output = stdout.decode(encoding, 'replace')


        # Output the tagged sentences
        limits = [0]
        passed = 0
        for i in range(len(lengths)):
          new = passed + lengths[i]
          limits.append(new)
          passed = new 
        tagged_sentences = senna_output.strip().split("\n\n")
        tagged_corpus = []
        for i in range(len(limits)-1):
          tagged_corpus.append(tagged_sentences[limits[i]:limits[i+1]])

        tagged_corpus = [self._process_article(article) for article in tagged_corpus]
        new_lengths = [len(article) for article in tagged_corpus]
        if new_lengths != lengths:
          raise Exception("The %d tagged sentences are not matching %d original"
                          " sentneces" % (sum(new_lengths), sum(lengths)))
        return tagged_corpus
