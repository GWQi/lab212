# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: audio.py
# Description: extract feature from an audio file
# *************************************************

import os
import librosa as lrs

# extract mbe features
from ASC.code.base import fparam

def extractMBE(path):
  """
  extract mbe feature
  param path : string, audio path
  return log_mbe : np.ndarray, features, shape=[mbe_order, T_frames]
  """
  signal, _ = lrs.load(path, sr=fparam.MBE_SAMPLE_RATE, mono=True)
  # log_mbe.shape=[mbe_order, T_frames]
  log_mbe = lrs.power_to_db(lrs.feature.melspectrogram(
                            y=signal,
                            sr=fparam.MBE_SAMPLE_RATE,
                            n_fft=int(fparam.MBE_FRAME_LENGTH*fparam.MBE_SAMPLE_RATE),
                            hop_length=int(fparam.MBE_FRAME_SHIFT*fparam.MBE_SAMPLE_RATE),
                            n_mels=fparam.MBE_ORDER)
                            )
  return log_mbe

def audio2MBE_inputs(path):
  """
  extract mbe from one audion and transform feature into inputs of the network
  param path : string, audio path
  return inputs : list of np.ndarray, cut the log mbe feature into slices, [array1, array2, ...], arrayn
  """
  inputs = []
  log_mbe = extractMBE(path)

  slices_num = int((log_mbe.shape[-1]-fparam.MBE_SEGMENT_LENGTH) / fparam.MBE_SEGMENT_SHIFT_TEST) + 1
  for i in list(range(slices_num)):
    inputs.append(log_mbe[:, i*fparam.MBE_SEGMENT_SHIFT_TEST:i*fparam.MBE_SEGMENT_SHIFT_TEST+fparam.MBE_SEGMENT_LENGTH])

  return inputs

