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
    self.segment_length = 1
    # default segment shift length, in second
    self.segment_shift = 0.5
    # default system sample rate, if the initial rate of wav file not 
    # equal to it, down/up-sample will be done during processing
    self.rate = 44100
    # weather use mfcc
    self.mfcc_on = True
    # weather use mel band energy
    self.mbe_on = False
    # mfcc order if use mfcc
    self.mfcc_order = 40
    # weather use delta mfcc
    self.mfcc_deta = True


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
                                                  -0001.dt
                                                  -0002.dt
                                                  -...
                                              -a002/
                                                  -0001.dt
                                                  -0002.dt
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
      self.cv_setup[i]["train_file"] = []
      for train_wav in self.cv_setup[i]["train"]:
        base_name = train_wav.split('.')[0]
        for filename in os.listdir(os.path.join(self.root_path, "development/data/{}/".format(base_name))):
          self.cv_setup[i]["train_file"].append(os.path.join(self.root_path, "development/data/{}/{}".format(base_name, filename)))


class DcaseTrainParam(object):
    """
    dcase dataset training parameters
    """

    def __init__(self):
        """
        training parameters defination and initialization
        """
        # batch size
        
      



    