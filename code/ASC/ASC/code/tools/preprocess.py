# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: preprocess.py
# Description: music/speech data preprocession, this file just be excuted once to for data oreoaration
# *************************************************

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_root)
sys.path.append(project_root)

import numpy as np
import librosa as lrs
from readlabel import readlabel
from ASC.code.base import fparam

def load(filepath, labelpath, category, sr):
  """
  load one category data
  """
  try:
    label = readlabel(labelpath)
  except FileNotFoundError as e:
    print("The {} file does not has corresponding label file!".format(filepath))
    return None

  if label.get(category, None) is None:
    print("The {} file doesn'y contain {} data! Please check the label file: {}".format(filepath, category, labelpath))
    return None

  data_all, _ = lrs.load(filepath, sr=sr, mono=True)

  data = np.zeros(0, dtype=data_all.dtype)

  for start, end in label[category]:
    data = np.append(data, data_all[int(start*sr) : int(end*sr+1)])

  return data

def main(musicroot, speechroot, fearoot):
  """
  extract features and cut, save to fearoot
  params:
    musicroot : string, music wav files root directory
    speechroot :
    fearoot : directory to save feature
  """

  label_content = ''

  # extract and cut music features
  for root, dirlist, filelist in os.walk(musicroot):
    for filename in filelist:
      if filename.endswith('.wav'):
        filepath = os.path.join(root, filename)
        labelpath = filepath.replace('.wav', '.lab')
        # cut feature of this file will saved under this directory
        feature_dir_relate = filepath.replace(musicroot, 'music').replace('.wav', '')
        feature_dir_abs = os.path.join(fearoot, feature_dir_relate)
        # make directory
        try:
          os.makedirs(feature_dir_abs)
        except:
          pass
        signal = load(filepath, labelpath, 'music', fparam.MBE_SAMPLE_RATE)
        # log_mbe.shape=[mbe_order, T_frames]
        log_mbe = lrs.power_to_db(lrs.feature.melspectrogram(
                                  y=signal,
                                  sr=fparam.MBE_SAMPLE_RATE,
                                  n_fft=int(fparam.MBE_FRAME_LENGTH*fparam.MBE_SAMPLE_RATE),
                                  hop_length=int(fparam.MBE_FRAME_SHIFT*fparam.MBE_SAMPLE_RATE),
                                  n_mels=fparam.MBE_ORDER)
                                  )
        # compute how many segment can be cutted
        cut_num = int((log_mbe.shape[1]-fparam.MBE_SEGMENT_LENGTH) / fparam.MBE_SEGMENT_SHIFT_TRAIN) + 1
        for i in list(range(cut_num)):
          np.save(os.path.join(feature_dir_abs, '{}.npy'.format(i+1)), log_mbe[:, i*fparam.MBE_SEGMENT_SHIFT_TRAIN : i*fparam.MBE_SEGMENT_SHIFT_TRAIN+fparam.MBE_SEGMENT_LENGTH])
          label_content += os.path.join(feature_dir_relate, '{}.npy'.format(i+1)) + "   " + 'music'
        
        print("{} file done!".format(filepath))


  # extract and cut speech features
  for root, dirlist, filelist in os.walk(speechroot):
    for filename in filelist:
      if filename.endswith('.wav'):
        filepath = os.path.join(root, filename)
        labelpath = filepath.replace('.wav', '.lab')
        # cut feature of this file will saved under this directory
        feature_dir_relate = filepath.replace(speechroot, 'speech').replace('.wav', '')
        feature_dir_abs = os.path.join(fearoot, feature_dir_relate)
        # make directory
        try:
          os.makedirs(feature_dir_abs)
        except:
          pass
        signal = load(filepath, labelpath, 'speech', fparam.MBE_SAMPLE_RATE)
        # log_mbe.shape=[mbe_order, T_frames]
        log_mbe = lrs.power_to_db(lrs.feature.melspectrogram(
                                  y=signal,
                                  sr=fparam.MBE_SAMPLE_RATE,
                                  n_fft=int(fparam.MBE_FRAME_LENGTH*fparam.MBE_SAMPLE_RATE),
                                  hop_length=int(fparam.MBE_FRAME_SHIFT*fparam.MBE_SAMPLE_RATE),
                                  n_mels=fparam.MBE_ORDER)
                                  )
        # compute how many segment can be cutted
        cut_num = int((log_mbe.shape[1]-fparam.MBE_SEGMENT_LENGTH) / fparam.MBE_SEGMENT_SHIFT_TRAIN) + 1
        for i in list(range(cut_num)):
          np.save(os.path.join(feature_dir_abs, '{}.npy'.format(i+1)), log_mbe[:, i*fparam.MBE_SEGMENT_SHIFT_TRAIN : i*fparam.MBE_SEGMENT_SHIFT_TRAIN+fparam.MBE_SEGMENT_LENGTH])
          label_content += os.path.join(feature_dir_relate, '{}.npy'.format(i+1)) + "   " + 'speech'
        print("{} file done!".format(filepath))

  with open(os.path.join(fearoot, 'label.txt'), 'w') as f:
    f.write(label_content)

if __name__ == '__main__':
  main('/media/guwenqi/Seagate Expansion Drive/ASC/train/music',
       '/media/guwenqi/Seagate Expansion Drive/ASC/train/speech',
       '/media/guwenqi/Seagate Expansion Drive/ASC/train/feature')