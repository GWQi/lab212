# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-21
# Email: hey_xiaoqi@163.com
# Filename: utils.py
# Description: this file contains some useful tools
# *************************************************
import numpy as np

# reference: sidekit
def compute_delta(features, win=3, method='filter', filt=numpy.array([.25, .5, .25, 0, -.25, -.5, -.25])):
  """
  features is a 2D-ndarray  each row of features is a a frame
    
  param features: the feature frames to compute the delta coefficients
  param win: parameter that set the length of the computation window. The size of the window is (win x 2) + 1
  param method: method used to compute the delta coefficients can be diff or filter
  param filt: definition of the filter to use in "filter" mode, default one is similar to SPRO4:  filt=numpy.array([.2, .1, 0, -.1, -.2])
        
  return: the delta coefficients computed on the original features.
  """
  # First and last features are appended to the begining and the end of the 
  # stream to avoid border effect
  x = numpy.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=NPDTYPE_DATA)
  x[:win, :] = features[0, :]
  x[win:-win, :] = features
  x[-win:, :] = features[-1, :]

  delta = numpy.zeros(x.shape, dtype=NPDTYPE_DATA)

  if method == 'diff':
    filt = numpy.zeros(2 * win + 1, dtype=NPDTYPE_DATA)
    filt[0] = -1
    filt[-1] = 1

  for i in range(features.shape[1]):
    delta[:, i] = numpy.convolve(features[:, i], filt)

  return delta[win:-win, :]


def reconstruct(rootdir, dstdir):
  """
  """