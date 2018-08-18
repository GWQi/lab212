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
                                              -constant/
                                                  -mean.npy
                                                  -std.npy
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

    # normalization constant
    self.mean = 0
    self.std = 1

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
    # concatenate all mbe to cumpute normalization constant
    mbe_all = np.zeros((self.feaparams.mbe_num,0))
    for wavfile in wavfiles:
      print("Extracting features form: {}".format(wavfile))
      basename = wavfile.split('.')[0]
      # first create directories
      try:
        os.makedirs(os.path.join(self.rootpath, "development/data/{}/segmented/".format(basename)))
      except:
        pass
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
                                       n_fft=int(self.feaparams.frame_length*self.feaparams.rate),
                                       hop_length=int(self.feaparams.frame_shift*self.feaparams.rate),
                                       power=2.0,
                                       n_mels=self.feaparams.mbe_num)
      # concatenate all mbe
      mbe_all = np.concatenate((mbe_all, mbe), axis=-1)
      # print (mbe.shape)
      labels = self.readLabels(os.path.join(self.rootpath, "development/meta/{}.ann".format(basename)),
                               mbe.shape[-1])
      # stick the labels on extracted features
      features = np.concatenate((labels, mbe))
      # write features on disk
      np.save(os.path.join(self.rootpath, "development/data/{}/{}".format(basename,basename)), features)
      print("Extracted features form: {}".format(wavfile))

    # compute normalization constant
    self.mean = mbe_all.mean(axis=-1).reshape(self.feaparams.mbe_num, -1)
    self.std = mbe_all.std(axis=-1).reshape(self.feaparams.mbe_num, -1)
    # save those constant onto disk
    try:
      os.makedirs(os.path.join(self.rootpath, "development/data/constant/"))
    except:
      pass
    np.save(os.path.join(self.rootpath, "development/data/constant/mean"), self.mean)
    np.save(os.path.join(self.rootpath, "development/data/constant/std"), self.std)

    return

  def cutFeatures(self):
    """
    cut features and save the cuted features on disk
    """
    wavfiles = os.listdir(os.path.join(self.rootpath, "development/audio/"))
    # load the normalization constant
    mean = np.load(os.path.join(self.rootpath, "development/data/constant/mean.npy"))
    std = np.load(os.path.join(self.rootpath, "development/data/constant/std.npy"))
    # first we remove the cutted files under segmented/ directory, then cut features and write them
    for wavfile in wavfiles:
      print ("cutting {} features".format(wavfile))
      basename = wavfile.split('.')[0]
      cut_dir = os.path.join(self.rootpath, "development/data/{}/segmented/".format(basename))
      shutil.rmtree(cut_dir)
      os.makedirs(cut_dir)
      # load the whole feature data and do normalization
      feadata = np.load(os.path.join(self.rootpath, "development/data/{}/{}.npy".format(basename,basename)))
      feadata[len(Dcase2017Task3Label):,:] = (feadata[len(Dcase2017Task3Label):,:] - mean) / std
      # cut and save cut-features according to sequence length and sequence shift
      cutnum = int((feadata.shape[-1]-self.feaparams.sequence_length)/self.feaparams.sequence_shift + 1)
      for i in range(0, cutnum):
        if i % 10 == 0:
          print ("{}'th cutted features, {} total!".format(i+1, cutnum))
        np.save(os.path.join(self.rootpath, "development/data/{}/segmented/{}".format(basename, str(i+1))),
               feadata[:,i*self.feaparams.sequence_shift : i*self.feaparams.sequence_shift+self.feaparams.sequence_length])

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
    labels = np.zeros((len(Dcase2017Task3Label), n_frames))

    # read label file
    with open(filepath, 'r') as f:
      for aline in f.readlines():
        content = aline.strip().split()
        # get the on-off time and compute start/end frame index
        start, end, label = float(content[2]), float(content[3]), content[4]
        # print (start, end, label)
        start = int(start/self.feaparams.frame_shift)
        end = min(int(end/self.feaparams.frame_shift), n_frames)
        # print (start, end)
        # arrange corresponding labels
        labels[:, start : end] +=\
        np.array([Dcase2017Task3Label[label] for i in range(end-start)]).transpose()

    return labels

  def cvConfigure(self):
    """
    cross validation configuration
    """
    self.tparam.cvConfigure(self.rootpath)
    return

  def fetchDataCV(self):
    """
    get one batch data and increase batch/fold counter
    return:
      x : np.ndarray, shape = (k_batchs, n_mbe, T_frame), training data
      y : np.ndarray, shape = (k_batchs, m_class, T_frame), labels
    note:
      if there has no data left to feath, both x and y are None and return
    """
    # increase batch counter and check weather has data left
    self.tparam.ith_batch += 1
    # check weather current fold has been ran out of
    if self.tparam.ith_batch*self.tparam.batch_size > len(self.tparam.cv_setup[self.tparam.kth_fold]["train_files"]):
      self.tparam.kth_fold += 1
      self.ith_batch = 1
    if self.tparam.kth_fold > self.tparam.KFold:
      # cross validation has been finished, no data left
      # reset kth_fold and ith_batch in case of next cross validation
      self.kth_fold = 1
      self.ith_batch = 0
      return None, None
    # fetch i'th batch training data in k'th fold
    x, y = [], []
    for filepath in self.tparam.cv_setup[self.tparam.kth_fold]["train_files"][(self.tparam.ith_batch-1)*self.tparam.batch_size : self.tparam.ith_batch*self.tparam.batch_size]:
      data = np.load(os.path.join(self.rootpath, filepath))
      x.append(data[len(Dcase2017Task3Label):, :])
      y.append(data[0:len(Dcase2017Task3Label), :])
    x = np.stack(x)
    y = np.stack(y)
    return x, y