# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-18
# Email: hey_xiaoqi@163.com
# Filename: post_procession.py
# Description: this file is used for post-procession
# *************************************************

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

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

  return content

def MBEProbs2speech_music_labels(probs, context=2):
  """
  apply median filter to probs and binary classification
  params:
    probs : list/tuple/np.array of probabilities
    context : int, window size is 1 + 2*context
  return content : string, label file content
  """
  # median smooth
  probs_smooth = medium_smooth(probs, context=context)
  # binary classification 
  binaries = np.where(probs_smooth > 0.5, 1, 0)
  # binary to labels
  labels = MBEBinary2speech_music_labels(binaries)

  return labels

def MBEProbs2speech_music_single(filepath, initial_probs, labelpath=None, context=2):
  """
  speech/music classification to one single file with initial probabilities
  params:
    filepath : string, path of the audio
    initial_probs : initial prediction probs
    labelpath : string, path where to store the labels
  """
  # smooth the probs, post-procession
  probs_smooth = medium_smooth(initial_probs)

  # classification
  binary = np.where(probs_smooth > 0.5, 1, 0)

  # if labelpath is None, plot the resault
  if labelpath is None:
    # load wav
    _, wav = wavfile.read(filepath)
    if wav.ndim == 2:
      wav = wav.mean(axis=-1)
    # plot wav, initial probs, smoothed probs, and classification resaults
    plt.figure(1, [6, 7])
    # plot the wav
    plt.subplot(411)
    plt.title("wav file")
    plt.plot(wav)

    # plot initial prediction probabilities
    plt.subplot(412)
    plt.title("Initial Probs")
    plt.plot(initial_probs)

    # plot smoothed probs
    plt.subplot(413)
    plt.title("Smooth Probs")
    plt.plot(probs_smooth)

    # plot final classification resaults
    plt.subplot(414)
    plt.title("classification")
    plt.plot(binary)
    return

  else:
    labels = MBEProbs2speech_music_labels(initial_probs, context=context)
    with open(labelpath, 'w') as f:
      f.write(labels)
    return
