#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

P_DATA = [10, 20, 50, 100]
P_DEV = [
0.380518570427,
0.416638254347,
0.457291805908,
0.499239065705]

P_TEST = [
0.372224,
0.406203,
0.465652,
0.502755]

P_TRAIN = [
0.930210595901,
0.896148456,
0.905014050045,
0.89585957999]

P_Lambda = [30, 100, 30, 30]
P_data_size = 150812
P_CFM = [
[8.5,  1.9,   2.1,   1.3,   1.2,   1.8],
[1.8,  7.9,  1.8,   1.6,   1.7,   1.9,],
[2.2,   1.9,  7.6,  1.8,   1.5,   1.6],
[1.2,   1.5,   1.5,  9.8,  1.3,   1.3],
[1.4,   2.0,   1.5,   1.4, 8.6,  1.8],
[1.8,   2.1,   1.8,   1.2,   1.9,  7.9]
]
P_Labels = ['Dutch', 'EN-US', 'German', 'Russian', 'Spanish', 'French']

G_DATA = [10, 20, 50, 100]
G_DEV = [
0.379146919431,
0.407659048796,
0.446489936181,
0.466518133432
]

G_TEST = [
0.367397,
0.390013,
0.440838,
0.475422
]

G_TRAIN = [
0.93693373601,
0.899991280844,
0.918121220451,
0.895290240458
]

G_CFM = [
[   9.4,   3.1,   2.5,   2.2,   2.8 ],
[   3.1,   8.3,   3.1,   3.3,   2.3 ],
[   3.0,   3.1,   8.5,  3.0,   2.5 ],
[   2.0,   2.6,   2.7,  10.4,  2.4 ],
[   2.6,   2.0,   2.3,   2.1, 11.0,]
]

G_Labels = ['EN-US', 'NGermanic', 'Roman', 'Uralic', 'Asian']

G_Lambda = [30, 100, 10, 100]
G_data_size = 81927


F_DATA = [10, 20, 50, 100]

F_DEV = [
0.6899603,
0.703526875096,
0.722650706485,
0.741967344671,
]

F_TEST = [
0.675663,
0.703800,
0.727078,
0.744497
]

F_TRAIN = [
0.966107,
0.954730787032,
0.944650623112,
0.942785442088
]

F_CFM = [
[37.6, 12.4],
[13.2, 36.8]
]
F_Labels = ['Non native', 'Native']

F_Lambda = [3, 100, 300, 300]
F_data_size = 322770

def main(options, args):
  fig = figure()
  ax1 = fig.add_subplot(212)
  ax1.grid(True)
  ylabel('Accuracy')
  p_dev = ax1.plot(P_DATA, P_DEV)
  p_test = ax1.plot(P_DATA, P_TEST, color='r')
  legend( (p_dev, p_test), ('Dev', 'Test'), 'upper left', shadow=True)
  ax2 = fig.add_subplot(211)
  ax2.grid(True)
  ylabel('Accuracy')
  p_train = ax2.plot(P_DATA, P_TRAIN)
  legend( (p_train), ('Training',), 'upper right', shadow=True)
  ax2.set_title('Learning Curves for Popular Languages Experiment')
  ax1.set_xlabel('Percentage of data')
  savefig('popular_lc.png', format='png')

  fig = figure()
  ax1 = fig.add_subplot(212)
  ax1.grid(True)
  ylabel('Accuracy')
  g_dev = ax1.plot(G_DATA, G_DEV)
  g_test = ax1.plot(G_DATA, G_TEST, color='r')
  legend( (p_dev, p_test), ('Dev', 'Test'), 'upper left', shadow=True)
  ax2 = fig.add_subplot(211)
  ax2.grid(True)
  ylabel('Accuracy')
  g_train = ax2.plot(G_DATA, G_TRAIN)
  legend( (g_train), ('Training',), 'upper right', shadow=True)
  ax2.set_title('Learning Curves for Languages Families Experiment')
  ax1.set_xlabel('Percentage of data')
  savefig('family_lc.png', format='png')

  fig = figure()
  ax1 = fig.add_subplot(212)
  ax1.grid(True)
  ylabel('Accuracy')
  f_dev = ax1.plot(F_DATA, F_DEV)
  f_test = ax1.plot(F_DATA, F_TEST, color='r')
  legend( (f_dev, f_test), ('Dev', 'Test'), 'upper left', shadow=True)
  ax2 = fig.add_subplot(211)
  ax2.grid(True)
  ylabel('Accuracy')
  f_train = ax2.plot(F_DATA, F_TRAIN)
  legend( (f_train), ('Training',), 'upper right', shadow=True)
  ax2.set_title('Learning Curves for Native vs Non-Native Writers Experiment')
  ax1.set_xlabel('Percentage of data')
  savefig('native_lc.png', format='png')

  fig = figure()
  title('Confusion Matrix for Popular Languages Experiment')
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(P_CFM), cmap=plt.cm.jet, interpolation='nearest')
  width = len(P_CFM)
  height = len(P_CFM[0])
  for x in xrange(width):
    for y in xrange(height):
      ax.annotate(str(P_CFM[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
  cb = fig.colorbar(res)
  xticks(range(width), P_Labels)
  yticks(range(height), P_Labels)
  savefig('popular_cfm.png', format='png')
  
  fig = figure()
  title('Confusion Matrix for Native Experiment')
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(F_CFM), cmap=plt.cm.jet, interpolation='nearest')
  width = len(F_CFM)
  height = len(F_CFM[0])
  for x in xrange(width):
    for y in xrange(height):
      ax.annotate(str(F_CFM[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
  cb = fig.colorbar(res)
  xticks(range(width), F_Labels)
  yticks(range(height), F_Labels)
  savefig('native_cfm.png', format='png')

  fig = figure()
  title('Confusion Matrix for Languages Families Experiment')
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(G_CFM), cmap=plt.cm.jet, interpolation='nearest')
  width = len(G_CFM)
  height = len(G_CFM[0])
  for x in xrange(width):
    for y in xrange(height):
      ax.annotate(str(G_CFM[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
  cb = fig.colorbar(res)
  xticks(range(width), G_Labels)
  yticks(range(height), G_Labels)
  savefig('family_cfm.png', format='png')

  show()

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

