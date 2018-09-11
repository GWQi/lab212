# Copyright 2018/7 by wenqi gu
# parameter class file
# author Wenqi Gu

import yaml
import os

from random import shuffle

class FeaParams(object):
  """
  this class define the parameters when extract features from wav file
  """

  def __init__(self):
    """
    parameters defination and initialization
    """

    # default frame length, in secong
    self.frame_length = 0.04
    # default frame shift length, in second
    self.frame_shift = 0.02
    # default segment length, in second
    self.sequence_length = 100
    # default segment shift length, in second
    self.sequence_shift = 30
    # default system sample rate, if the initial rate of wav file not 
    # equal to it, down/up-sample will be done during processing
    self.rate = 44100
    # weather mono wav
    self.mono = True
    # weather use mfcc
    self.mfcc_on = False
    # weather use mel band energy
    self.mbe_on = True
    self.mbe_num = 40
    # mfcc order if use mfcc
    self.mfcc_order = 40
    # weather use delta mfcc
    self.mfcc_deta = False


  def Configure_dict(self, cfg):
    """
    parameters configuration

    params:
      cfg : dict, key-value pair of feature extracting parameters
    """

    if not cfg.get("frame_length"):
      self.frame_length = cfg['frame_length']

    if not cfg.get('frame_shift'):
      self.frame_shift = cfg['frame_shift']

    if not cfg.get('segment_length'):
      self.segment_length = cfg['segment_length']

    if not cfg.get('segment_shift'):
      self.segment_shift = cfg['segment_shift']

    if not cfg.get('rate'):
      self.rate = cfg['rate']

    if not cfg.get('mfcc_on'):
      self.mfcc_on = cfg['mfcc_on']

    if not cfg.get('mbe_on'):
      self.mbe_on = cfg['mbe_on']

    if not cfg.get('mfcc_order'):
      self.mfcc_order = cfg['mfcc_order']

    if not cfg.get('mfcc_deta'):
      self.mfcc_deta = cfg['mfcc_deta']

    return

  def Configure_file(self, cfg):
    """
    parameters configuration

    params:
      cfg : string, yaml file path
    """

    with open(cfg, 'r') as f:
      cfg = yaml.load(f)
    self.Configure_dict(cfg)

    return

class DcaseTrainParam(object):
  """
  dcase dataset training parameters
  """

  def __init__(self):
    """
    training parameters defination and initialization
    """
    # batch size
    self.batch_size = 50
    # K-fold
    self.KFold = 5
    # k'th fold and i'th batch used to checkout
    self.ith_batch = 0
    self.kth_fold = 1

    # number of epoch, when cross-validation is finished , use all train data to train final model
    # this paramter will be used then
    # flag indicates that weather begin to train the final model
    self.final_train_ = 0
    self.num_epoch_ = 5
    # used to checkout
    self.kth_epoch = 0
    # cross-validation setup, the files path in cv_setup should be shuffled before training
    """
    cv_setup = {
                1 : {"train" : ["a001.wav", "a002.wav", ...], 
                     "test"  : ["b001.wav", "b002.wav", ...],
                     "train_files" : ["development/data/a001/1.npy", "development/data/a123/23.npy", ...],
                     "test_files"  : ["development/data/b001/1.npy", "development/data/b023/31.npy", ...]
                    }
                2 : {"train" : ["a005.wav", "a006.wav", ...], 
                     "test"  : ["b003.wav", "b007.wav", ...],
                     "train_files" : ["development/data/a005/1.npy", "development/data/a13/23.npy", ...],
                     "test_files"  : ["development/data/b011/1.npy", "development/data/b013/31.npy", ...]
                    }

                ...

                5 : ...
               }
    """
    self.cv_setup = {}
    # weather save check point, if never save checkpoint, reset {ith_batch, kth_fold, cv_setup}
    self.saved_check = False

  def setBatchSize(self, size):
    """
    set batch size
    params:
      size : int, new batch size  
    """
    self.batch_size = size

  def saveCheckPoint(self, filepath):
    """
    this function is used to save self to disk
    param:
      filepath : string, path to save checkpoint
    """

    # set saved_check True
    self.saved_check = True
    pass

  def cvConfigure(self, rootpath):
    """
    cross validation configuration
    params:
      rootpath : str, dataset directory rootpath
    """
    if len(self.cv_setup) != 0:
      return
    # load cross-validation files partion
    with open(os.path.join(rootpath, "development/evaluation_setup/evaluation.yaml"), 'r') as f:
      self.cv_setup = yaml.load(f)

    self.KFold = len(self.cv_setup)
    # configure cross validation train_files/test_files
    for i in range(1, self.KFold+1):
      # configure train_files
      self.cv_setup[i]["train_files"] = []
      for wav_file in self.cv_setup[i]["train"]:
        basename = wav_file.split('.')[0]
        for cut_file in os.listdir(os.path.join(rootpath, "development/data/{}/segmented/".format(basename))):
          self.cv_setup[i]["train_files"].append("development/data/{}/segmented/{}".format(basename, cut_file))
      # shuffle the train files order
      shuffle(self.cv_setup[i]["train_files"])

      # configure test_files
      self.cv_setup[i]["test_files"] = []
      for wav_file in self.cv_setup[i]["test"]:
        basename = wav_file.split('.')[0]
        for cut_file in os.listdir(os.path.join(rootpath, "development/data/{}/segmented/".format(basename))):
          self.cv_setup[i]["train_files"].append("development/data/{}/segmented/{}".format(basename, cut_file))

    return
