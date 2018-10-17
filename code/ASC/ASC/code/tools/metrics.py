# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-10-15
# Email: hey_xiaoqi@163.com
# Filename: metrics.py
# Description: this file is used for system testing
# *************************************************
import os
import sys
import numpy as np

ASC_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ASC_PROJECT_ROOT not in sys.path:
  sys.path.append(ASC_PROJECT_ROOT)

from ASC.code.base import universe

# ***********************************
# * Those is used for music/speech classification
# ************************************

def label2array_truth(label_path, resolution=0.01):
  """
  transform truth labels to np.array, elements in the
  array represent weather the sample is music or speech
  params:
    label_path : string, label file path
    resolution : float, time resolution in second
  return:
    array : np.ndarray, shape=(time,)
    start : float, start time of this label file
    end   : float, end time of this label file
  """
  with open(label_path, 'r') as f:
    lines = f.readlines()
    start = float(lines[0].strip().split()[0])
    end = float(lines[-1].strip().split()[0])

    num_decisions = int(end/resolution) - int(start/resolution) + 1

    array = np.ones(num_decisions, dtype=np.int32)

    for linenum, aline in enumerate(lines):
      start_, end_, label = aline.strip().split()

      array[int((float(start_)-start)/resolution) : int((float(end_)-start)/resolution)] =\
      universe.SPEECH_MUSIC_DIC.get(label, 1)
    return array, start, end

def label2array_pred(label_path, resolution, start, end):
  """
  transform prediction labels to np.array, elements ine the array
  represent weather the sample is music or speech
  params:
    label_path : string, label file path
    resolution : float, time resolution in second
    start : float, start time of the corresponding truth label file
    end   : float, end time of the corresponding truth label file
  return:
    array : np.ndarray, shape=(time,)
  """
  with open(label_path, 'r') as f:
    lines = f.readlines()
    start_pred = float(lines[0].strip().split()[0])
    end_pred = float(lines[-1].strip().split()[0])

    num_decisions = int(end_pred/resolution) - int(start_pred/resolution) + 1
    array = np.ones(num_decisions, dtype=np.int32)

    for linenum, aline in enumerate(lines):
      start_, end_, label = aline.strip().split()
      
      array[int((float(start_)-start_pred)/resolution) : int((float(end_)-start_pred)/resolution)] =\
      universe.SPEECH_MUSIC_DIC.get(label, 1)

    return array[int(start_pred/resolution)-int(start/resolution) :\
                 int(min(end, end_pred)/resolution)-int(start/resolution)]

def precision(truth_labels, pred_labels):
  """
  compute classification precision
  params:
    truth_labels : directory where store the ground truth labels
    pred_labels : directory where store the predictions
  return:
    truth_rate : precision
    error_rate : error rate
  """
  all_classification_num = 0
  all_classification_error = 0
  for root, dirlist, filelist in os.walk(truth_labels):
    for filename in filelist:
      if filename.endswith('.lab'):
        truth_label_path = os.path.join(root, filename)
        pred_label_path = truth_label_path.replace(truth_labels, pred_labels)
        # truth label array
        truth_array, start, end = label2array_truth(truth_label_path, resolution=0.01)
        pred_array = label2array_pred(pred_label_path, start, end)

        valid_compare_samples = min(truth_array.size, pred_array.size)

        all_classification_num += valid_compare_samples
        all_classification_error += (truth_array[:valid_compare_samples] != pred_array[:valid_compare_samples]).sum()

  error_rate = all_classification_error*1.0 / all_classification_num
  truth_rate = 1 - error_rate
  return truth_rate, error_rate
