# Copyright 2018/8 by wenqi gu
# dcase dataset front-end processing, feature extraction
# author Wenqi Gu

import librosa
import os
import numpy as np

from SED.project.src.params.param import FeaParams
from SED.project.src.utils.env import Dcase2017Task3Label

class DCASEData2017Task2(object):
  """
  this class used to extract features
  """
  def __init__(self):
    """
    Attributes defination
    """
    # feature extracting params
    self.feaparams = FeaParams()

  def FeaParamConf_file(self, cfg):
    """
    feature params configuration
    params:
      cfg : string, configuration files, yaml format
    """
    self.feaparams.Configure_file(cfg)

  def FeaParamConf_dict(self, cfg):
    """
    feature params configuration
    params:
      cfg : dict, configuration key-value pairs
    """
    self.feaparams.Configure_dict(cfg)

  def ExtractFeatures(self, rootpath):
    """
    feature extraction
    params:
      rootpath : string, dataset directory path, dcase dataset directory structure must satify:
                                  -root/
                                      -development/
                                          -audio/
                                              -a001.wav
                                              -a002.wav
                                              -...
                                          -data/
                                              -a001/
                                                  -a001.dt
                                                  -segmented/
                                                      -0001.dt
                                                      -0002.dt
                                                      -...
                                              -a002/
                                                  -a002.dt
                                                  -segmented/
                                                      -0001.dt
                                                      -0002.dt
                                                      -...
                                      -evaluation/
                                          -audio/
                                              -b001.wav
                                              -b002.wav
                                              -...
                                          -meta/
                                              -b001.ann
                                              -b002.ann
                                              -...
    """
    # traverse wav files under audio/ and create corresponding directory under data/
    wavfiles = os.listdir(os.path.join(rootpath, "development/audio/"))
    for wavfile in wavfiles:
      os.makedirs(os.path.join(rootpath, "development/data/{}/segmented/".format(wavfile.split('.')[0])))

  def readLabels(self, filepath, n_frames):
    """
    read dcase data annotation file
    params:
      filepath : string, file path of label
    return:
      labels : np.ndarray, shape=(k_class, n_frames) contains 0/1, 
                    [
                    [1, 1, 0, 0, 0, 0], this is the first frame indicate the brakes and car events happens at this frame
                    [0, 1, 0, 0, 0, 0],
                    ...
                    ]
    """
    # 6 class, brakes, car, children, vehicle, speak, walk
    labels = np.zeros(6, n_frames)

    # read label file
    with open(filepath, 'r') as f:
      for aline in f.readlines():
        content = aline.strip().split()
        start, end, label = content[]






