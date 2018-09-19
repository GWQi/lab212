# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-18
# Email: hey_xiaoqi@163.com
# Filename: post_procession.py
# Description: this file is used for post-procession
# *************************************************

import numpy as np

def medium_smooth(probs, context=2):
  """
  apply medium filter to smooth the probs
  params:
    probs : list/tuple/np.array of probabilities
    context : int, window size is 1 + 2*context
  return:
    prob : np.array, smoothed probabilities
  """
  probs = np.asarray(probs, dtype=np.float32)
  # pad edge values
  probs_pad = np.pad(probs, [context, context], 'edge')

  for i in list(range(probs.size)):
    probs[i] = probs_pad[i:i+2*context+1].mean()

  return probs

def median_smooth(classifications, context=2):
  """
  apply median filter to smooth the classification resaults
  params:
    classifications : list/tuple/np.array of classification resaults
    context : int, window size is 1 + 2*context
  """
  classifications = np.asarray(classifications, dtype = np.int32)
  # pad edge values
  classifications_pad = np.pad(classifications, [context, context], 'edge')

  for i in list(range(classifications.size)):
    classifications[i] = classifications_pad[i:i+2*context+1].median()

  return classifications