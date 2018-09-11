# -*- coding: utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np

# get the lab212 code root path and append it to sys.path
CODEROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(CODEROOT)


def createFlags():
  