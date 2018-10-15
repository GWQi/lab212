# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-18
# Email: hey_xiaoqi@163.com
# Filename: post_procession.py
# Description: this file is used for post-procession
# *************************************************
import sys
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
    context : smoothed context size, int, window size is 1 + 2*context
  return:
    prob : np.array, smoothed probabilities
  """
  probs = np.asarray(probs, dtype=np.float32)
  # pad edge values
  probs_pad = np.pad(probs, [context, context], 'edge')

  for i in list(range(probs.size)):
    probs[i] = probs_pad[i:i+2*context+1].mean()

  return probs

def medium_smooth_(probs, context=int(fparam.MBE_SEGMENT_LENGTH/fparam.MBE_SEGMENT_SHIFT_TEST)):
  """
  medium filter to smooth the probs
  params:
    probs : list/tuple/np.array of probabilities, probs in cumputed by tensorflow
            the length of probs is the segments number
    context : smoothed context size, only consider the left context
  return:
    tiny_probs : smoothed probs, this probs is for every segment shift time duration 
  """
  probs = np.asarray(probs, dtype=np.float32)

  num_segments = probs.size

  # compute how many tiny decision is made by the classifier, the tiny decision is made for every resolution time duration
  num_tiny_decision = (num_segments-1) + context

  # initialize the tiny probs for every tiny decision
  tiny_probs = np.zeros(num_tiny_decision, dtype=np.float32)

  for i in list(range(tiny_probs.size)):
    tiny_probs[i] = probs[max(0, i-context//2) : min(num_tiny_decision, i+context//2)].mean()

  return tiny_probs

def median_smooth(classifications, context=2):
  """
  apply median filter to smooth the classification resaults
  params:
    classifications : list/tuple/np.array of classification resaults
    context : smoothed context size, int, window size is 1 + 2*context
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
  resolution = fparam.MBE_SEGMENT_SHIFT_TEST * fparam.MBE_FRAME_SHIFT
  binaries = np.asarray(binaries, dtype=np.int32)
  content = ""
  start = 0
  for idx in list(range(1, binaries.size)):
    if binaries[idx] != binaries[idx-1]:
      content += "%.3f %.3f %s\n" % (start*resolution,
                                    (idx+1)*resolution,
                                    SPEECH_MUSIC_DIC_R[binaries[idx-1]])
      start = idx+1

  if binaries[-1] == binaries[-2]:
    content += "%.3f %.3f %s\n" % (start*resolution,
                                  (binaries.size)*resolution,
                                  SPEECH_MUSIC_DIC_R[binaries[-1]])
  return content

def MBEProbs2speech_music_labels(probs, context=2):
  """
  apply median filter to probs and binary classification
  params:
    probs : list/tuple/np.array of probabilities
    context : smoothed context size, int, window size is 1 + 2*context
  return content : string, label file content
  """
  # median smooth
  probs_smooth = medium_smooth_(probs)

  # binary classification 
  binaries = np.where(probs_smooth > 0.5, 1, 0)
  # binary to labels
  labels = MBEBinary2speech_music_labels(binaries)

  return labels

def MBEProbs2speech_music_single(filepath, initial_probs, labelpath="", context=2):
  """
  speech/music classification to one single file with initial probabilities
  params:
    filepath : string, path of the audio
    initial_probs : initial prediction probs
    labelpath : string, path where to store the labels
    context : smoothed context size, int, window size is 1 + 2*context
  """

  # if labelpath is "", plot the resault
  if labelpath == "":
    # smooth the probs, post-procession
    probs_smooth = medium_smooth_(initial_probs)

    # classification
    binary = np.where(probs_smooth > 0.5, 1, 0)

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

    plt.show()
    return

  else:
    labels = MBEProbs2speech_music_labels(initial_probs, context=context)
    with open(labelpath, 'w') as f:
      f.write(labels)
    return
