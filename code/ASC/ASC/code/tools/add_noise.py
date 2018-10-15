# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:2018-10-8
# Email: hey_xiaoqi@163.com
# Filename: add_noise.py
# Description: add noise 
# *************************************************

import sys
import math
import random
import argparse

import numpy as np
import librosa as lrs
from scipy.io import wavfile
from vad import vad



def add_noise(srcfile, noisefile, sr, segment=5.0, SNR=0):
  """
  add noise file on srcfile by SNR
  params:
    srcfile : string, source wav file path
    noisefile : string, noise file path
    sr: int, resample rate
    segment : float, time duration in seconds
    SRN : float
  return
    signal : data after adding noise

  Note: we use librosa to load data, so the noised signal will be 32bit
        both source wav file and noise wav file will resample to sr,
        and noised data will be convert to mono
  """

  src_data, _ = lrs.load(srcfile, sr=sr, mono=True)
  src_data = vad(src_data, sr=sr, percent=20.0)
  noise_data, _ = lrs.load(noisefile, sr=sr, mono=True)

  # if noise data is longer than source data, then truncate noise data
  if src_data.size < noise_data.size:
    offset = random.randint(0, noise_data.size-src_data.size)
    noise_data = noise_data[offset:offset+src_data.size]
  elif src_data.size > noise_data.size:
    noise_data = np.concatenate([noise_data]*int(src_data.size/noise_data.size+1))
    offset = random.randint(0, noise_data.size-src_data.size)
    noise_data = noise_data[offset:offset+src_data.size]
  else:
    pass

  # scale factors array, source file factors, 1-scale is noise file factors
  scale = np.zeros((src_data.size), dtype=src_data.dtype)

  # segment length in samples
  segment = int(sr*segment)

  # compute scale factors every segment
  # 10*log((alpha**2*S)/(beta**2*N)) = SNR   (1)
  # alpha + beta = 1                         (2)
  # we compute alpha and beta according (1) and (2)
  for i in list(range(int(src_data.size/segment + 1))):
    S = (src_data[i*segment:(i+1)*segment]**2).mean()
    N = (noise_data[i*segment:(i+1)*segment]**2).mean()
    beta = 1.0 / (1.0 + math.sqrt(10**(SNR/10.0) * N/S))
    alpha = 1 - beta
    scale[i*segment:(i+1)*segment] = alpha

  filter_size = 11
  filt = np.ones(filter_size) / filter_size
  win = int((filter_size-1) / 2)
  scale = np.pad(scale, (win, win), mode='edge')

  scale = np.convolve(scale, filt, mode='valid')

  signal = scale*src_data + (1-scale) * noise_data
  signal = signal.astype(dtype=np.float32)

  return signal

def write_add_noise(srcfile, dstfile, noisefile, sr, segment=10.0, SNR=0):
  """
  add noise file on srcfile by SNR
  params:
    srcfile : string, source wav file path
    noisefile : string, noise file path
    sr: int, resample rate
    segment : float, time duration in seconds
    SRN : float
  return
    signal : data after adding noise

  Note: we use librosa to load data, so the noised signal will be 32bit
        both source wav file and noise wav file will resample to sr,
        and noised data will be convert to mono
  """

  # add noise
  signal = add_noise(srcfile, noisefile, sr, segment, SNR=SNR)

  # write wav file
  wavfile.write(dstfile, sr, signal)

  return

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--src", type=str, default="", help="source file path")
  parser.add_argument("-n", "--noise", type=str, default="", help="noise file path")
  parser.add_argument("-d", "--dst", type=str, default="", help="noised file path")
  parser.add_argument("-r", "--rate", type=int, default=16000, help="sample rate while adding noise")
  parser.add_argument("--snr", type=float, default=0.0, help="signal noise ratio")

  args = parser.parse_args()

  if args.src == "":
    print("Please specipy the source file path")
    sys.exit(-1)
  if args.noise == "":
    print("Please specipy the noise file path")
    sys.exit(-1)
  if args.dst == "":
    print("Please specipy the output file path")
    sys.exit(-1)

  write_add_noise(args.src, args.dst, args.noise, args.rate, 5.0, SNR=args.snr)



if __name__ == "__main__":
  main()