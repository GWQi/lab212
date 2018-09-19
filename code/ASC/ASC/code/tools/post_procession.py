# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-18
# Email: hey_xiaoqi@163.com
# Filename: post_procession.py
# Description: this file is used for post-procession
# *************************************************

import numpy as np

from ASC.code.base import fparam
from ASC.code.base.universe import SPEECH_MUSIC_DIC_R

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

def MBEBinary2speech_music_labels(binaries):
  """
  given binaries of music/speech classification, get its corresponding label content
  param binary : binary classification
  return content : string, label file content
  """
  binaries = np.asarray(binaries, dtype=np.int32)
  content = ""
  start = 0
  for idx in list(range(1, binaries.size)):
    if binaries[idx] != binaries[idx-1]:
      content += "{} {} {}\n".format(start*fparam.MBE_SEGMENT_SHIFT_TEST*fparam.MBE_FRAME_SHIFT,
                                     idx*fparam.MBE_SEGMENT_SHIFT_TEST*fparam.MBE_FRAME_SHIFT,
                                     SPEECH_MUSIC_DIC_R[binaries[idx-1]])
      start = idx