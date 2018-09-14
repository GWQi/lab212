# -*- coding: utf-8 -*-
import os
def readlabel(srcfile_path):
    """
    版 本 号：0.0.1
  
    修改日期：2017-11-03
  
     函数功能：从给定的源label文件中读打好的标签，并将打好的标签进行分类
 
     参    数：
            srcfile_path: label文件路径
  
     返 回 值：
             label: 字典类型，其键值是label标签的类别，每个键值对应的value是该类标签对应的音频时间段；
                      label = { 
                               'class1' : [[start1, end1], [start2, end2], ...],
                               'class2' : [[start1, end1], [start2, end2], ...],
                               ...
                               'classN' : [[start1, end1], [start1, end2], ...]
                              }
    """
  
    label = {}

    # 判断label文件路径是否存在，若不存在则输出哪里错了然后退出脚本
    if not os.path.exists(srcfile_path):
        raise FileNotFoundError()

    with open(srcfile_path, 'r') as labelfile:
        for aline in labelfile.readlines():
            #datastr是label文件中每行split之后的列表，datastr[0]为这段语音的起始时间，
            #datastr[1]为这段语音的结束时间，datastr[2]为这段语音的相应的label
            datastr = aline.strip().split()
          
            #判断这行的label类型是否已经存在，若存在则直接append，不存在则先创建相应的键值，并将此行的起止时间作为初始值为键值赋值
            if datastr[2] in label:
                label[datastr[2]].append([float(datastr[0]), float(datastr[1])])
            else:
                label[datastr[2]] = [[float(datastr[0]), float(datastr[1])]]

    return label