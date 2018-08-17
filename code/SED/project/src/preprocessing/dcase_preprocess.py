# Copyright 2018/8 by wenqi gu
# dcase dataset front-end processing, feature extraction
# author Wenqi Gu

import os
import shutil
import numpy as np
import librosa as lrs

from SED.project.src.params.param import FeaParams, DcaseTrainParam
from SED.project.src.utils.env import Dcase2017Task3Label

class DCASEData2017Task2(object):
  """
  this class used to extract features
  """
  def __init__(self, rootpath):
    """
    Attributes defination
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
                                                  -a001.npy
                                                  -segmented/
                                                      -0001.npy
                                                      -0002.npy
                                                      -...
                                              -a002/
                                                  -a002.npy
                                                  -segmented/
                                                      -0001.npy
                                                      -0002.npy
                                                      -...
                                          -evaluation_setup/
                                              -evaluation.yaml
                                          -meta/
                                              -a001.ann
                                              -a002.ann
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
    # feature extracting params
    self.feaparams = FeaParams()
    # training parameters
    self.tparam = DcaseTrainParam()
    # 
    self.rootpath = rootpath

  def feaParamConf_file(self, cfg):
    """
    feature params configuration
    params:
      cfg : string, configuration files, yaml format
    """
    self.feaparams.Configure_file(cfg)

  def feaParamConf_dict(self, cfg):
    """
    feature params configuration
    params:
      cfg : dict, configuration key-value pairs
    """
    self.feaparams.Configure_dict(cfg)

  def extractFeatures(self):
    """
    feature extraction
    """
    # traverse wav files under audio/ and create corresponding directory under data/
    wavfiles = os.listdir(os.path.join(self.rootpath, "development/audio/"))
    for wavfile in wavfiles:
      basename = wavfile.split('.')[0]
      # first create directories
      os.makedirs(os.path.join(self.rootpath, "development/data/{}/segmented/".format(basename)))
      # load data and compute mel band energy, 40 band
      data, _ = lrs.load(os.path.join(self.rootpath, "development/audio/{}".format(wavfile)),
                         sr=self.feaparams.rate,
                         mono=self.feaparams.mono)
      #************************************************************
      # now we extract features, if we want to change the net topology
      # or change the feature we use, just change the codes here
      #************************************************************
      mbe = lrs.feature.melspectrogram(data,
                                       sr=self.feaparams.rate,
                                       n_fft=self.feaparams.frame_length*self.feaparams.rate,
                                       hop_length=self.feaparams.frame_shift*self.feaparams.rate,
                                       power=2.0,
                                       n_mels=40)
      labels = self.readLabels(os.path.join(self.rootpath, "development/meta/{}.ann".format(basename)),
                               mbe.shape[-1])
      # stick the labels on extracted features
      features = np.concatenate(labels, mbe)
      # write features on disk
      np.save(os.path.join(self.rootpath, "development/data/{}/{}".format(basename,basename)), features)

    return

  def cutFeatures(self):
    """
    cut features and save the cuted features on disk
    """
    wavfiles = os.listdir(os.path.join(self.rootpath, "development/audio/"))
    # first we remove the cutted files under segmented/ directory, then cut features and write them
    for wavfile in wavfiles:
      basename = wavfile.split('.')[0]
      cut_dir = os.path.join(self.rootpath, "development/data/{}/segmented/".format(basename))
      shutil.rmtree(cut_dir)
      os.makedirs(cut_dir)
      # load the whole feature data
      feadata = np.load(os.path.join(self.rootpath, "development/data/{}/{}.npy".format(basename,basename)))
      # cut and save cut-features according to sequence length and sequence shift
      for i in range(0, int((feadata.shape[-1]-self.feaparams.sequence_length)/self.feaparams.sequence_shift + 1)):
        np.save(os.path.join(self.rootpath, "development/data/{}/segmented/{}".format(basename, str(i+1))),
               feadata[:,i*self.feaparams.sequence_shift, i*self.feaparams.sequence_shift+self.feaparams.sequence_length])

    return

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
        # get the on-off time and compute start/end frame index
        start, end, label = content[2], content[3], content[4]
        start = int(start/self.feaparams.frame_shift)
        end = int(end/self.feaparams.frame_shift)
        # arrange corresponding labels
        labels[:, start : end+1] +=\
        np.array([Dcase2017Task3Label[label] for i in range(end-start+1)]).transpose()

    return labels