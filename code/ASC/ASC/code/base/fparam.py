# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: fparam.py
# Description: feature params setting
# *************************************************

# **********************************************
# * mbe feature setting, global feature setting
# **********************************************

# mbe order
MBE_ORDER = 32
# frame length in second
MBE_FRAME_LENGTH = 0.04
# frame shift in second
MBE_FRAME_SHIFT = 0.02
# mbe context length in frames
MBE_SEGMENT_LENGTH = 100
# mbe context shift in frames while training model
MBE_SEGMENT_SHIFT_TRAIN = 33
# mbe context shift in frames in test phase
MBE_SEGMENT_SHIFT_TEST = 5
# mbe model operating sample rate
MBE_SAMPLE_RATE = 16000

# **********************************************
# * mfcc feature setting, global feature setting
# **********************************************

# mfcc order
MFCC_ORDER = 26