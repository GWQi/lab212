# Copyright 2018 by wenqi gu
# parameter class file
# author Wenqi Gu

class FeaParam(object):
  """
  this class define the parameters when extract features from wav file
  """

  def __init__(self):
    """
    parameters defination and initialization
    """

    # default frame length, in secong
    self.frame_length_ = 0.04
    # default frame shift length, in second
    self.frame_shift_ = 0.02
    # default segment length, in second
    self.segment_length_ = 1
    # default segment shift length, in second
    self.segment_shift_ = 0.5
    # default system sample rate, if the initial rate of wav file not 
    # equal to it, down/up-sample will be done during processing
    self.rate_ = 16000
    # weather use mfcc
    self.mfcc_on_ = True
    # mfcc order if use mfcc
    self.mfcc_order_ = 40


  def Configure_(self, )