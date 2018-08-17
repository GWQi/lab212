# Copyright 2018/7 by wenqi gu
# parameter class file
# author Wenqi Gu

import yaml
import os

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
    self.mfcc_on = True
    # weather use mel band energy
    self.mbe_on = False
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

class DataPrep(object):
  """
  dataset preproration
  """

  def __init__(self, dirpath):
    """
    training parameters defination and initialization

    params:
      dirpath : string, dataset directory path, dcase dataset directory structure must satify:
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
    # dataset directory path
    self.root_path = dirpath
    # i'th fold cross-validation, 0 indicates that training not start
    self.ith_fold = 0
    # cross validation setup, 
    # self.cv_setup = {1 : {"train" : ["a001.wav", "a002.wav", ...], "test" : ["b001.wav", "b002.wav"]}, 2 : {...}, ...}
    with open(os.path.join(self.root_path, "development/evaluation_setup/evaluation.yaml"), 'r') as f:
      self.cv_setup = yaml.load(f)
    # K-fold
    self.Kfold = len(self.cv_setup)
    # configure cross validation training files
    for i in range(1,self.Kfold+1):
      self.cv_setup[i]["train_files"] = []
      for train_wav in self.cv_setup[i]["train"]:
        base_name = train_wav.split('.')[0]
        for filename in os.listdir(os.path.join(self.root_path, "development/data/{}/".format(base_name))):
          self.cv_setup[i]["train_files"].append(os.path.join(self.root_path, "development/data/{}/{}".format(base_name, filename)))


class DcaseTrainParam(object):
  """
  dcase dataset training parameters
  """

  def __init__(self):
    """
    training parameters defination and initialization
    """
    # batch size
    self.batch_size_ = 50
    # K-fold
    self.Kfold_ = 5
    # k'th fold and i'th batch used to checkout
    self.ith_batch = 0
    self.kth_fold = 0

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
    self.batch_size_ = size

  def checkPoint(self):
    """
    this function is used to save self to disk
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
    if len(cv_setup) != 0:
      return
    # load cross-validation files partion
    with open(os.path.join(self.root_path, "development/evaluation_setup/evaluation.yaml"), 'r') as f:
      self.cv_setup = yaml.load(f)

    self.Kfold_ = len(self.cv_setup)
    # configure cross validation train_files/test_files
    for i in range(1, self.Kfold_+1):
      self.


      



    