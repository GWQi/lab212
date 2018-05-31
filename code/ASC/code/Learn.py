# -*- coding: utf-8 -*-
#/python3.5
import librosa as lrs
import pandas as pd
import numpy as np
import scipy.fftpack as fft

import getopt
import os
import builtins
import math
import sys
import logging
import gc
import copy
import pickle

from tools.readlabel import readlabel
from Config import Config
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ShuffleSplit
from scipy import stats

class Power(object):
    """
    all kinds of thresholds such as extreme speech(music) and corresponding I, Er.
    
    params:
        name : string, name of feature statistics
        speech : np.ndarray, shape=(n_segments,), the feature statistics of speech audio
        music : np.ndarray, shape=(m_segments,), the feature statistics of music audio
        logger : training logger instance
        free : bool, default is Ture, weather to free speech and music statistics and corresponding
               kde after everything is done.

    attributes(vars):
        name_ : string, the name of this feature statistics
        speech_ : np.ndarray, shape=(n_segments), speech statistics data
        music_ : np.ndarray, shape=(m_segments), music statistics data

        _speech_kde : the kernel density function estimator of speech statistics
        _music_kde : the kernel density function estimator of music statistics
    
        _logger : logger instance must be freed after everything is done, otherwise the instance of Power can't be writed on disk by pickle
        _speech_position :
        _music_position : 

        extreme_speech_left_ :
        extreme_speech_right_ :
        extreme_music_left_ = :
        extreme_music_right_ :

        high_speech_ :
        high_music_ :
        
        separation_ :

        es_inclusion_ :
        es_error_ :
        em_inclusion_ :
        em_error_ :
        hs_inclusion_ :
        hs_error_ :
        hm_inclusion_ :
        hm_error_ :

    attributes(methods):
        

    """
    def __init__(self, name, speech, music, logger, free=True):
        super(Power, self).__init__()

        self._free = free
        self._logger = logger
        self.name_ = name
        self.speech_ = speech
        self.music_ = music

        # model the speech, music statistics using gaussian kernel function
        self._speech_kde = stats.gaussian_kde(self.speech_)
        self._music_kde = stats.gaussian_kde(self.music_)

        speech_mean = self.speech_.mean()
        music_mean = self.music_.mean()

        if speech_mean < music_mean:
            self._speech_position = 'left'
            self._music_position = 'right'
        else:
            self._speech_position = 'right'
            self._music_position = 'left'

        self.extreme_speech_left_ = -np.inf
        self.extreme_speech_right_ = np.inf
        self.extreme_music_left_ = -np.inf
        self.extreme_music_right_ = np.inf

        self.high_speech_ = None
        self.high_music_ = None

        self.separation_ = None

        self.es_inclusion_ = None
        self.es_error_ = 0
        self.em_inclusion_ = None
        self.em_error_ = 0
        self.hs_inclusion_ = None
        self.hs_error_ = None
        self.hm_inclusion_ = None
        self.hm_error_ = None

        self.es_power_ = None
        self.em_power_ = None
        self.hs_power_ = None
        self.hm_power_ = None
        self.sp_power_ = None

        # compute the thresholds and inclusion(error) fraction
        self.init()


    def init(self):

        # compute all kinds of thresholds
        self._thresholds()

        # compute all kinds of inclusion(error) fraction
        self._inclusion_error()

        # compute the separation power for every threshold
        self._power()

        # free the data
        if self._free:
            # free speech data
            del self.speech_
            gc.collect()
            self.speech_ = None

            # free music data
            del self.music_
            gc.collect()
            self.music_ = None

            # free speech kde
            del self._speech_kde
            gc.collect()
            self._speech_kde = None

            # free music kde
            del self._music_kde
            gc.collect()
            self._music_kde = None

            # free logger
            del self._logger
            gc.collect()
            self._logger = None



    def _thresholds(self):
        """
        compute the extreme(high) speech(music) thresholds
        """
        speech_max = self.speech_.max()
        speech_min = self.speech_.min()
        music_max = self.music_.max()
        music_min = self.music_.min()

        # ***********************compute the extreme thresholds***********************
        if speech_max > music_max:
            self.extreme_speech_right_ = music_max
        else:
            self.extreme_music_right_ = speech_max

        if speech_min < music_min:
            self.extreme_speech_left_ = music_min
        else:
            self.extreme_music_left_ = speech_min

        # *******************compute the high probabilyty thresholds**************
        # compute the probility of number between min(speech_min, music_min) and max(speech_max, music_max), step length is (max(speech_max, music_max) - min(speech_min, music_min)) / 1000
        numbers = np.arange(min(speech_min, music_min), max(speech_max, music_max), (max(speech_max, music_max) - min(speech_min, music_min)) / 1000)
        speech_prob = self._speech_kde(numbers)
        music_prob = self._music_kde(numbers)

        # compute the prob difference between speech and music by subtracting music_prob from speech_prob
        prob_diff = speech_prob - music_prob

        # the max value of prob_diff is high probility speech threshold point where where the difference between the
        # height of the speech PDF and the height of the music PDF is maximal; the min value of prob_diff is high
        # probility music threshold point where where the difference between the height of the music PDF and the
        # height of the speech PDF is maximal
        self.high_speech_ = numbers[prob_diff.argmax()]
        self.high_music_ = numbers[prob_diff.argmin()]

        # ********************compute the separation point**************
        if speech_max < music_min:
            self.separation_ = (speech_max + music_min) / 2
        elif music_max < speech_min:
            self.separation_ = (music_max + speech_min) / 2
        else:
            # find the point where the joint decision error is the smallest
            joint_decision_errors = np.zeros(len(numbers))
            for i in range(len(numbers)):
                if i%100 == 0:
                    self._logger.info("Computing {}'th bin joint dicision error for feature: {}, {} totally!".format(i, self.name_, len(numbers)))
                joint_decision_errors[i] = self._joint_dicision_error(numbers[i])
            self.separation_ = numbers[joint_decision_errors.argmin()]

        return

    def _inclusion_error(self):
        """
        compute inclusion fraction and error fraction for extreme(high) speech(music) thresholds

        Er: the percentage of incorrect segments that exceed the threshold. For speech thresholds
        these are the music segments, and for music thresholds these are the speech segments.
        Note that by the definition of the extreme thresholds, their error fractions are 0.
        """
        self.es_inclusion_ = np.where([self.speech_ < self.extreme_speech_left_, self.speech_ > self.extreme_speech_right_], 1, 0).mean() * 2
        self.em_inclusion_ = np.where([self.music_ < self.extreme_music_left_, self.music_ > self.extreme_music_right_], 1, 0).mean() * 2

        # two cases, one case: the speech statistics pdf is on the left, music ststistics pdf in on the right;
        #            anonther case: the speech statistics pdf is on the right, music ststistics pdf in on the left
        if self._speech_position == 'left':
            self.hs_inclusion_ = np.where(self.speech_ <= self.high_speech_, 1, 0).mean()
            self.hs_error_ = np.where(self.music_ <= self.high_speech_, 1, 0).mean()
            self.hm_inclusion_ = np.where(self.music_ >= self.high_music_, 1, 0).mean()
            self.hm_error_ = np.where(self.speech_ >= self.high_music_, 1, 0).mean()

        else:
            self.hs_inclusion_ = np.where(self.speech_ >= self.high_speech_, 1, 0).mean()
            self.hs_error_ = np.where(self.music_ >= self.high_speech_, 1, 0).mean()
            self.hm_inclusion_ = np.where(self.music_ <= self.high_music_, 1, 0).mean()
            self.hm_error_ = np.where(self.music_ <= self.high_music_, 1, 0).mean()

        return

    def _power(self):
        """
        compute the separation power for every threshold
        """
        # compute extreme speech(music) thresholds' separation power
        self.es_power_ = self.es_inclusion_
        self.em_power_ = self.em_inclusion_

        # compute the high probability speech(music) thresholds' separation power.
        self.hs_power_ = self.hs_inclusion_**2 / (self.hs_error_ + 0.000001)
        self.hm_power_ = self.hm_inclusion_**2 / (self.hm_error_ + 0.000001)

        # compute the separation threshold separation power
        self.sp_power_ = (self.speech_.mean() - self.music_.mean())**2 / (self.speech_.std()**2 + self.music_.std()**2)

        return



    def _joint_dicision_error(self, a):
        """
        compute the joint decision error at a

        params:
            a : float, separation point

        return:
            joint_decision_error : float, joint decision error
        """

        if self._speech_position == 'left':
            # use the log probability instead of directly probability
            # here may be changed, change np.inf ot max(self._speech_max,
            # self._music_max), change -np.inf to min(self._speech_min, self._music_min)
            speech_decision_error = self._speech_kde.integrate_box_1d(a, np.inf)
            music_decision_error = self._music_kde.integrate_box_1d(-np.inf, a)
        else:
            speech_decision_error = self._speech_kde.integrate_box_1d(-np.inf, a)
            music_decision_error = self._music_kde.integrate_box_1d(a, np.inf)

        joint_decision_error = 0.5 * (speech_decision_error + music_decision_error)

        return joint_decision_error

    def update_extreme_threshold(self, speech, music):
        """
        update extreme speech/music threshold using new data.

        parameters:
            speech : np.ndarray, shape=(n_samples,) new speech data
            music : np.ndarray, shape=(m_samples,), new music data
        """
        speech_max = speech.max()
        speech_min = speech.min()
        music_max = music.max()
        music_min = music.min()

        
        if speech_max > music_max:
            self.extreme_speech_right_ = music_max
        else:
            self.extreme_music_right_ = speech_max

        if speech_min < music_min:
            self.extreme_speech_left_ = music_min
        else:
            self.extreme_music_left_ = speech_min

    def update_high_probability_threshold(self, speech, music):
        """
        update high probability speech/music threshold using new data.

        parameters:
            speech : np.ndarray, shape=(n_samples,) new speech data
            music : np.ndarray, shape=(m_samples,), new music data
        """
        speech_max = speech.max()
        speech_min = speech.min()
        music_max = music.max()
        music_min = music.min()

        # compute the probility of number between min(speech_min, music_min) and max(speech_max, music_max), step length is (max(speech_max, music_max) - min(speech_min, music_min)) / 1000
        numbers = np.arange(min(speech_min, music_min), max(speech_max, music_max), (max(speech_max, music_max) - min(speech_min, music_min)) / 1000)
        
        # fit music/speech data using gaussian kernel function
        speech_kde = stats.gaussian_kde(speech)
        music_kde = stats.gaussian_kde(music)

        speech_prob = speech_kde(numbers)
        music_prob = music_kde(numbers)

        # compute the prob difference between speech and music by subtracting music_prob from speech_prob
        prob_diff = speech_prob - music_prob

        # the max value of prob_diff is high probility speech threshold point where where the difference between the
        # height of the speech PDF and the height of the music PDF is maximal; the min value of prob_diff is high
        # probility music threshold point where where the difference between the height of the music PDF and the
        # height of the speech PDF is maximal
        self.high_speech_ = numbers[prob_diff.argmax()]
        self.high_music_ = numbers[prob_diff.argmin()]

    def update_separation_threshold(self, speech, music):
        """
        update high speech/music separation threshold using new data.

        parameters:
            speech : np.ndarray, shape=(n_samples,) new speech data
            music : np.ndarray, shape=(m_samples,), new music data
        """
        speech_max = speech.max()
        speech_min = speech.min()
        music_max = music.max()
        music_min = music.min()

        # fit music/speech data using gaussian kernel function
        speech_kde = stats.gaussian_kde(speech)
        music_kde = stats.gaussian_kde(music)

        # bins over which to search separation threshold
        numbers = np.arange(min(speech_min, music_min), max(speech_max, music_max), (max(speech_max, music_max) - min(speech_min, music_min)) / 1000)

        if speech_max < music_min:
            self.separation_ = (speech_max + music_min) / 2
        elif music_max < speech_min:
            self.separation_ = (music_max + speech_min) / 2
        else:

            def joint_decision_error(a, speech_position):
                if speech_position == 'left':
                    speech_decision_error = speech_kde.integrate_box_1d(a, np.inf)
                    music_decision_error = music_kde.integrate_box_1d(-np.inf, a)
                else:
                    speech_decision_error = speech_kde.integrate_box_1d(-np.inf, a)
                    music_decision_error = music_kde.integrate_box_1d(a, np.inf)

                joint_decision_error_ = 0.5 * (speech_decision_error + music_decision_error)

                return joint_decision_error_

            # find the point where the joint decision error is the smallest
            joint_decision_errors = np.zeros(len(numbers))
            for i in range(len(numbers)):
                joint_decision_errors[i] = joint_decision_error(numbers[i], self._speech_position)
            self.separation_ = numbers[joint_decision_errors.argmin()]





        
class Train(builtins.object):
    """
    return a instance that can caculate kinds of audio features,
    this class just can handle with 16000hz and mono wav files.
    
    parameters:
        cfg : string, config file path

    attributes(vars):
        _with_statistics : 
        _config : 
        _data : 
        _frame_length : 
        _frame_hop_length : 
        _segment_length : 
        _segment_hop_length : 
        _nframes_asegment : 
        _n_fft : 
        _mfcc_order : 
        _roll_percent : 
        _statistics_column_values : 


    """
    def __init__(self, cfg):
        """
        Parameters:
            cfg : string; config file path. set frame length, frame shift, segment length and so on
        """
        # config inition
        self.init(cfg)

        # setting parameters initialization
        self._operating_rate = self._config['operating_rate']
        self._frame_length = int(self._config['frame_size'] * self._config['operating_rate'])
        self._frame_hop_length = int(self._config['frame_shift'] * self._config['operating_rate'])
        self._segment_length = int(self._config['segment_size'] * self._config['operating_rate'])
        self._segment_hop_length = int(self._config['segment_shift'] * self._config['operating_rate'])
        self._n_fft = self._config['n_fft']
        self._mfcc_order = self._config['mfcc_order']
        self._roll_percent = self._config['roll_percent']

        # check whether the parameters are reasonable
        if self._frame_length % self._frame_hop_length != 0:
            raise ValueError('frame_length must can be divisible by frame_hop_length, please check and modify the config file!')

        if self._segment_length % self._segment_hop_length != 0:
            raise ValueError('segment_length must can be divisible by segment_hop_length, please check and modify the config file!')            
 
        if (self._segment_length - self._frame_length) % self._frame_hop_length == 0:
            self._nframes_asegment = int(1 + (self._segment_length - self._frame_length) / self._frame_hop_length)
        else:
            raise ValueError('a segment must can be splited into integer number of frames, please check and modify the config file!')

        if (self._segment_hop_length - self._frame_length) % self._frame_hop_length == 0:
            self._nframes_asegment_hop = int(1 + (self._segment_hop_length - self._frame_length) / self._frame_hop_length)
        else:
            raise ValueError('a segment hop must can be splited into integer number of frames, please check and modify the config file!')

        # load data, automatically resample the data to operating rate, and convert audio signal to mono if it's stero. 
        # self._data, _ = lrs.load(src, sr=self._operating_rate, mono=True, dtype=np.float64)
        self._data=None

        # statistics columns names
        self._statistics_column_values = ['ste_mean', 'ste_std', 'ste_mean_diff', 'ste_std_diff',
                                           'zcr_mean', 'zcr_std', 'zcr_skewness', 'zcr_mean_diff', 'zcr_std_diff', 'zcr_skewness_diff',
                                           'lber_mean', 'lber_std', 'lber_mean_diff', 'lber_std_diff',
                                           'hber_mean', 'hber_std', 'hber_mean_diff', 'hber_std_diff',
                                           'corr_mean', 'corr_std', 'corr_mean_diff', 'corr_std_diff']
        self._statistics_column_values.extend(['mfcc_mean_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['mfcc_std_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['mfcc_mean_diff_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['mfcc_std_diff_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['mfccdn_mean_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['mfccdn_std_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['mfccdn_mean_diff_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['mfccdn_std_diff_{}'.format(i) for i in range(1, self._mfcc_order+1)])
        self._statistics_column_values.extend(['rolloff_mean', 'rolloff_std', 'rolloff_mean_diff', 'rolloff_std_diff',
                                               'centroid_mean', 'centroid_std', 'centroid_mean_diff', 'centroid_std_diff',
                                               'flux_mean', 'flux_std', 'flux_mean_diff', 'flux_std_diff',
                                               'spread_mean', 'spread_std', 'spread_mean_diff', 'spread_std_diff',
                                               'lster'])


        if not self._with_statistics in ['True', 'true']:
            self._write_all_statistics()
        
        self._write_feature_ranking()
        self._cross_validation()



    def init(self, cfg):
        """
        Parameters:
            cfg : string; config file path. set frame length, frame shift, segment length and so on
        """
        # load the config file
        self._config = Config(cfg).cfgdic

        # check if has setted those three basic parameters in config file
        if not self._config.get('frame_size', None) or\
           not self._config.get('frame_shift', None) or\
           not self._config.get('segment_size', None) or\
           not self._config.get('segment_shift', None):
            raise ValueError('Please set basic parameters in config file: frame_size, frame_shift, segment_size, segment_shift')
        else:
            self._config['frame_size'] = float(self._config['frame_size'])
            self._config['frame_shift'] = float(self._config['frame_shift'])
            self._config['segment_size'] = float(self._config['segment_size'])
            self._config['segment_shift'] = float(self._config['segment_shift'])

        # set the system's operating sample rate, default is 16000
        if not self._config.get('operating_rate', None):
            self._config['operating_rate'] = 16000
        else:
            self._config['operating_rate'] = int(self._config['operating_rate'])

        if not self._config.get('mfcc_order', None):
            raise ValueError('Please set the order of mfcc in config file!')
        else:
            self._config['mfcc_order'] = int(self._config['mfcc_order'])

        if not self._config.get('roll_percent', None):
            raise ValueError('Please set the percent of roll off frequence in config file!')
        else:
            self._config['roll_percent'] = float(self._config['roll_percent'])

        if not self._config.get('n_fft', None):
            raise ValueError('Please set the number of fft points in config file!')
        else:
            self._config['n_fft'] = int(self._config['n_fft'])

        # train model using audio data or statistics data
        self._with_statistics = self._config.get('with_statistics', None)
        if self._with_statistics not in ['True', 'true']:
            # get train, save files paths and save name and save format
            if not self._config.get('music_files_dir', None):
                raise ValueError('Please set the music directory path!')
            else:
                if not os.path.exists(self._config['music_files_dir']):
                    raise ValueError('The music directory path:{} does not exist, please set a correct path!'.format(self._config['music_files_dir']))
                else:
                    self._music_files_dir = self._config['music_files_dir']

            if not self._config.get('speech_files_dir', None):
                raise ValueError('Please set the speech directory path!')
            else:
                if not os.path.exists(self._config['speech_files_dir']):
                    raise ValueError('The speech directory path:{} does not exist, please set a correct path!'.format(self._config['speech_files_dir']))
                else:
                    self._speech_files_dir = self._config['speech_files_dir']

        if not self._config.get('statistics_files_dir', None):
            raise ValueError('Please set the statistics directory path!')
        else:
            if not os.path.exists(self._config['statistics_files_dir']):
                os.makedirs(self._config['statistics_files_dir'])
            self._statistics_files_dir = self._config['statistics_files_dir']

        # get statistics files basename, if None, default is statistics
        self._music_statistics_name = self._config.get('music_statistics_name', 'music')
        self._speech_statistics_name = self._config.get('speech_statistics_name', 'speech')

        # if train using statistics data, must check the file path is existing
        if self._with_statistics in ['True', 'true']:
            if not (os.path.exists(os.path.join(self._statistics_files_dir, self._music_statistics_name+'.csv'))) or\
               not (os.path.exists(os.path.join(self._statistics_files_dir, self._speech_statistics_name+'.csv'))):
                raise ValueError("Train model using statistics, but the statistic files doesn't exist, Please check the statistics files path!")

        # get statistics files format, if None, default is csv and mat('cm', c is csv, m is mat)
        self._statistics_format = self._config.get('statistics_format', 'cm')


        if not self._config.get('model_files_dir', None):
            raise ValueError('Please set the model files directory path!')
        else:
            if not os.path.exists(self._config['model_files_dir']):
                os.makedirs(self._config['model_files_dir'])
            self._model_files_dir = self._config['model_files_dir']

        # get model file basename, if None, default is model
        self._model_name = self._config.get('model_name', 'model')

        # get the weighting factors determin the contributions of separation power and mutual correlation
        self._alpha = float(self._config.get('alpha', 0.5))
        self._beta = float(self._config.get('beta', 0.5))
        if (self._alpha+self._beta) != 1.0:
            raise ValueError("The sum of weighting factors(alpha, beta) determin the contributions of separation power and mutual correlation must be 1, please correct the cfg file!")

        # self._data, _ = lrs.load(src, sr=self._operating_rate, mono=True, dtype=np.float64)
        # get log file path, and initialize the logger
        if self._config.get('log_file_path', None) is None:
            raise ValueError("Please set log file path in the config file!")
        else:
            log_file_path = self._config['log_file_path']
            log_dir = os.path.split(log_file_path)[0]
            if os.path.exists(log_dir):
                self._logfile = log_file_path
            else:
                raise ValueError("The logging file directory: {} does not exist, please give a valid logging file path".format(log_dir))
        # logger configuration
        logging.basicConfig(filename=self._logfile,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            filemode='w',
                            level=logging.DEBUG)
        self._logger = logging.getLogger('Training procession')



    def _load(self, src, lab, category, **argv):
        """
        load one kind category data.(such as speech, music or ...)
        """

        try:
            label = readlabel(lab)
        except FileNotFoundError as e:
            self._logger.error("The {} file does not has corresponding label file!".format(lab))
            return None

        # here is supposed to have a logging, log this errro to a log file
        if label.get(category, None) is None:
            self._logger.warn("The {} file doesn'y contain {} data! Please check the label file: {}".format(src, category, lab))
            return None

        try:
            data_all, _ = lrs.load(src, **argv)
        except:
            self._logger.error("Load {} wav data failed, please check the file!".format(src))
            return None

        data = np.zeros(0, dtype=np.float64)
        for start, end in label[category]:
            data = np.append(data, data_all[int(start*self._operating_rate) : int(end*self._operating_rate+1)])

        return data


    def _short_time_energy(self):
        """
        caculate short time energy for each frame
        
        Returns:
            ste : np.ndarray [shape=(n_frames,)]; ste[i] is the fraction of short time energy in the i th frame
        """

        framed_data = lrs.util.frame(self._data,\
                                     frame_length=self._frame_length,\
                                     hop_length=self._frame_hop_length).transpose()
        ste = 10 * np.log10(1.0 / self._frame_length * np.sum(framed_data**2, axis=-1) + 1)

        return ste

    def _zero_crossing_rate(self):
        """
        caculate zero crossing rate for each frame

        Returns:    
            zcr : np.ndarray [shape=(n_frames,)]; zcr[i] is the fraction of zero crossings in the i th frame

        """
        zcr = lrs.feature.zero_crossing_rate(self._data,\
                                             frame_length=self._frame_length,\
                                             hop_length=self._frame_hop_length,\
                                             center=False)[0]
        return zcr

    def _band_energy_ratio(self):
        """
        caculate low band energy ratio and high band energy ratio. low band is below 10Hz, high band is above (4+int(sr/7))kHz

        Returns:    
            low : np.ndarray [shape=(n_frames,)]; low[i] is the fraction of low band energy ratio in the i th frame
            high : np.ndarray [shape=(n_frames,)]; high[i] is the fraction of high band energy ratio in the i th frame

        """
        # short time fft matrix

        # set low band boundary and high band boundary
        low_band_bound = 70
        high_band_bound = 4000 + int(self._operating_rate / 7)
        stft_matrix = self._stft()
        stft_square = stft_matrix**2

        total_energy = np.sum(stft_square[:,1:]) + 0.0000001 # avoid being diveded by zero
        low_energy = np.sum(stft_square[:, 1 : int(low_band_bound/self._operating_rate*self._n_fft)], axis=-1)
        high_energy = np.sum(stft_square[:, int(high_band_bound/self._operating_rate*self._n_fft):], axis=-1)

        low = 10 * np.log10(low_energy / total_energy + 0.0000001)
        high = 10 * np.log10(high_energy / total_energy + 0.0000001)

        return low, high

    def _autocorrelation_coefficient(self):
        """
        caculate the local peak of the normalized autocorrelation sequence of the frame

        returns:
            peak : numpy.ndarray, shape=(n_frames,), peak[i] is the autocorrelation local peak value of the i'th frame.
        """

        # give the bound of autocorrelation
        m_1 = int(0.003 * self._operating_rate)
        m_2 = int(0.016 * self._operating_rate)

        # data framing
        framed_data = lrs.util.frame(self._data,\
                                     frame_length=self._frame_length,\
                                     hop_length=self._frame_hop_length).transpose()

        # caculate the normalization factors for each frame
        norm_factors = np.sum(framed_data**2, axis=-1) + 0.0000001

        # pre-allocate the peak array
        peak = np.sum(framed_data[:, 0 : self._frame_length-m_1] * framed_data[:, m_1:], axis=-1)

        for m in range(m_1+1, m_2+1):
            peak = np.maximum(np.sum(framed_data[:, 0 : self._frame_length-m] * framed_data[:, m:], axis=-1), peak)
        
        # normalization
        peak = peak / norm_factors

        return peak

    def _mfcc(self):
        """
        compute the n-order MFC coefficients for each frame and MFCC difference vector between neighboring frames

        returns:
            mfcc : np.ndarray, shape=(n_frames, i); mfcc[n,i] is the i'th order MFC coefficient of n'th frame
            mfcc_diff_norm : np.ndarray, shape = (n_frames-1, i); mfcc_diff_norm[n, i] is the difference of i'th
                             order MFC coefficient between n+1 and n'th frame.

        """
        mfcc = lrs.feature.mfcc(self._data, sr=self._operating_rate,\
                                n_mfcc=self._mfcc_order,\
                                n_fft=self._frame_length,\
                                hop_length=self._frame_hop_length).transpose()

        mfcc_diff = mfcc[1:, :] - mfcc[0 : -1, :]
        mfcc_diff_norm_factor = np.linalg.norm(mfcc_diff, ord=2, axis=-1, keepdims=True) #2-norm, Euclidean norm
        mfcc_diff_norm = mfcc_diff / (mfcc_diff_norm_factor + 0.0000001)

        return mfcc, mfcc_diff_norm

    def _spectrum_rolloff(self):
        """
        compute roll-off frequency for each frame

        returns:
            roll_off ; np.ndarray, shape=(n_frames,); roll_off[i] is the roll-off frequence of i'th frame
        """
            
        roll_off = lrs.feature.spectral_rolloff(self._data,\
                                                sr=self._operating_rate,\
                                                n_fft=self._frame_length,\
                                                hop_length=self._frame_hop_length,\
                                                roll_percent=self._roll_percent)[0]

        return roll_off

    def _spectrum_centroid(self):
        """
        compute the spectral centroid for each frame.

        returns:
            centroid : np.ndarray, shape=(n_frames,); centroid[i] is the centroid frequence of i'th frame
        """
        centroid = lrs.feature.spectral_centroid(self._data,\
                                                 sr=self._operating_rate,\
                                                 n_fft=self._frame_length,\
                                                 hop_length=self._frame_hop_length)[0]
        return centroid

    def _spectral_flux(self):
        """
        compute the spectrum fluctuations between two consecutive audio frames

        returns:
            flux : np.ndarray, shape=(n_frames-1,); flux[n]is the spectral flux between n+1 and n'th frame
        """
        stft_matrix = self._stft()

        flux_matrix = stft_matrix[1:,:] - stft_matrix[0:-1,:]

        flux_matrix = flux_matrix**2

        flux = np.sum(flux_matrix, axis=-1)

        return flux

    def _spectrum_spread(self):
        """
        compute how the spectrum is concentrated around the perceptually adapted audio spectrum centroid

        returns:
            spread : np.ndarray, shape=(n_frames,)
        """
        # compute fft for each frame and discard the 0 frequence, stft_matrix.shape = (n_frames,n_fft/2)
        stft_matrix = self._stft()[:,1:]

        stft_square = stft_matrix**2

        # compute the sum of fft square for each frame, plus 0.0000001 to avoid computing error
        stft_square_sum = np.sum(stft_square, axis=-1) + 0.0000001

        # compute the center frequencies of each frequence bin and discard 0 frequency
        frequencies_bin = lrs.core.fft_frequencies(sr=self._operating_rate, n_fft=self._n_fft)[1:]

        # compute the perceptually adapted audio spectral centroid for each frame
        ASC = np.sum(np.log2(frequencies_bin/1000 * stft_square + 0.0000001), axis=-1) / stft_square_sum

        # compute spectrum spread
        spread = np.sqrt(np.sum((np.log2(frequencies_bin/1000) - ASC.reshape(-1,1))**2 * stft_square, axis=-1) / stft_square_sum)

        return spread

    def _short_time_energy_statistics(self):
        """
        compute statistical parameters of short time energy

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the short time energy across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the short time energy across the segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of short time
                        energy between consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of short
                       time energy between consecutive analysis points
        """
        ste = self._short_time_energy()
        mean, std, _, mean_diff, std_diff, __ = self._feature_statistics_helper_one(ste)

        return mean, std, mean_diff, std_diff

    def _zero_crossing_rate_statistics(self):
        """
        compute statistical parameters of short time energy

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the zero crossing rate across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the zero crossing rate across the segment
            skewness : 
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of zero crossing
                        rate between consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of short
                       zero crossing rate consecutive analysis points
            skewness_diff : 
        """
        zcr = self._zero_crossing_rate()
        mean, std, skewness, mean_diff, std_diff, skewness_diff = self._feature_statistics_helper_one(zcr, skew=True)

        return mean, std, skewness, mean_diff, std_diff, skewness_diff

    def _band_energy_ratio_statistics(self):
        """
        compute statistical parameters of low and high band energy ratio

        returns:
            low_mean : np.ndarray, shape=(n_segments,); mean value of the low band energy ratio across the segment
            low_std : np.ndarray, shape=(n_segments,); standard deviation of the low band energy ratio across the segment
            low_mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of low band energy
                            ratio between consecutive analysis points
            low_std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of low band
                           energy ratio between consecutive analysis points
            high_mean : np.ndarray, shape=(n_segments,); mean value of the high band energy ratio across the segment
            high_std : np.ndarray, shape=(n_segments,); standard deviation of the high band energy ratio across the segment
            high_mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of high band energy
                             ratio between consecutive analysis points
            high_std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of high band
                            energy ratio between consecutive analysis points
        """
        low_band_ratio, high_band_ratio = self._band_energy_ratio()

        low_mean, low_std, _, low_mean_diff, low_std_diff, __ = self._feature_statistics_helper_one(low_band_ratio)

        high_mean, high_std, _, high_mean_diff, high_std_diff, __ = self._feature_statistics_helper_one(high_band_ratio)

        return low_mean, low_std, low_mean_diff, low_std_diff, high_mean, high_std, high_mean_diff, high_std_diff

    def _autocorrelation_coefficient_statistics(self):
        """
        compute statistical parameters of autocorrelation coefficient

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the autocorrelation coefficient across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the autocorrelation coefficient across the segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of autocorrelation coefficient
                        between consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of autocorrelation
                       coefficient consecutive analysis points
        """
        coeff_peak = self._autocorrelation_coefficient()

        mean, std, _, mean_diff, std_diff, __ = self._feature_statistics_helper_one(coeff_peak)

        return mean, std, mean_diff, std_diff

    def _mfcc_statistics(self):
        """
        compute statistical parameters of mfcc

        returns:
            mfcc_mean : np.ndarray, shape=(n_segments, self._mfcc_order); mean value of the mfcc across the segment
            mfcc_std : np.ndarray, shape=(n_segments, self._mfcc_order); standard deviation of the mfcc across the segment
            mfcc_mean_diff : np.ndarray, shape=(n_segments, self._mfcc_order); mean value of the difference magnitude of mfcc
                             between consecutive analysis points
            mfcc_std_diff : np.ndarray, shape=(n_segments, self._mfcc_order); standard deviation of the difference magnitude
                            of mfcc consecutive analysis points
            mfcc_diff_norm_mean : np.ndarray, shape=(n_segments, self._mfcc_order)
            mfcc_diff_norm_std : np.ndarray, shape=(n_segments, self._mfcc_order)
            mfcc_diff_norm_mean_diff : np.ndarray, shape=(n_segments, self._mfcc_order)
            mfcc_diff_norm_std_diff : np.ndarray, shape=(n_segments, self._mfcc_order)
        """
        mfcc, mfcc_diff_norm = self._mfcc()

        # because each MFC coefficient is considered as one independ feature, so mfcc matrix and mfcc_diff_norm feature matrix
        # must be handled one column by one column, and then append them togather
        mfcc_mean, mfcc_std, _, mfcc_mean_diff, mfcc_std_diff, __ = self._feature_statistics_helper_one(mfcc[:, 0])
        mfcc_diff_norm_mean, mfcc_diff_norm_std, mfcc_diff_norm_mean_diff, mfcc_diff_norm_std_diff = self._feature_statistics_helper_two(mfcc_diff_norm[:, 0])
       
        # reshape to be appended
        mfcc_mean = mfcc_mean.reshape(-1,1)
        mfcc_std = mfcc_std.reshape(-1,1)
        mfcc_mean_diff = mfcc_mean_diff.reshape(-1,1)
        mfcc_std_diff = mfcc_std_diff.reshape(-1,1)

        mfcc_diff_norm_mean = mfcc_diff_norm_mean.reshape(-1,1)
        mfcc_diff_norm_std = mfcc_diff_norm_std.reshape(-1,1)
        mfcc_diff_norm_mean_diff = mfcc_diff_norm_mean_diff.reshape(-1,1)
        mfcc_diff_norm_std_diff = mfcc_diff_norm_std_diff.reshape(-1,1)

        # handle the mfcc and mfcc_diff_matrix one column by one column
        for i in range(1, self._mfcc_order):
            mfcc_mean_, mfcc_std_, _, mfcc_mean_diff_, mfcc_std_diff_, __ = self._feature_statistics_helper_one(mfcc[:, i])
            mfcc_diff_norm_mean_, mfcc_diff_norm_std_, mfcc_diff_norm_mean_diff_, mfcc_diff_norm_std_diff_ = self._feature_statistics_helper_two(mfcc_diff_norm[:, i])

            mfcc_mean = np.append(mfcc_mean, mfcc_mean_.reshape(-1,1), axis=-1)
            mfcc_std = np.append(mfcc_std, mfcc_std_.reshape(-1,1), axis=-1)
            mfcc_mean_diff = np.append(mfcc_mean_diff, mfcc_mean_diff_.reshape(-1,1), axis=-1)
            mfcc_std_diff = np.append(mfcc_std_diff, mfcc_std_diff_.reshape(-1,1), axis=-1)

            mfcc_diff_norm_mean = np.append(mfcc_diff_norm_mean, mfcc_diff_norm_mean_.reshape(-1,1), axis=-1)
            mfcc_diff_norm_std = np.append(mfcc_diff_norm_std, mfcc_diff_norm_std_.reshape(-1,1), axis=-1)
            mfcc_diff_norm_mean_diff = np.append(mfcc_diff_norm_mean_diff, mfcc_diff_norm_mean_diff_.reshape(-1,1), axis=-1)
            mfcc_diff_norm_std_diff = np.append(mfcc_diff_norm_std_diff, mfcc_diff_norm_std_diff_.reshape(-1,1), axis=-1)

        return mfcc_mean, mfcc_std, mfcc_mean_diff, mfcc_std_diff, mfcc_diff_norm_mean, mfcc_diff_norm_std, mfcc_diff_norm_mean_diff, mfcc_diff_norm_std_diff

    def _spectrum_rolloff_statistics(self):
        """
        compute statistical parameters of spectrum rolloff

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the spectrum rolloff across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the spectrum rolloff across the segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of spectrum rolloff
                        between consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of spectrum
                       rolloff consecutive analysis points
        """

        rolloff = self._spectrum_rolloff()

        mean, std, _, mean_diff, std_diff, __ = self._feature_statistics_helper_one(rolloff)

        return mean, std, mean_diff, std_diff

    def _spectrum_centroid_statistics(self):
        """
        compute statistical parameters of spectrum centroid

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the spectrum centroid across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the spectrum centroid across the segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of spectrum centroid
                        between consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of spectrum
                       centroid consecutive analysis points
        """
        centroid = self._spectrum_centroid()

        mean, std, _, mean_diff, std_diff, __ = self._feature_statistics_helper_one(centroid)

        return mean, std, mean_diff, std_diff

    def _spectral_flux_statistics(self):
        """
        compute statistical parameters of spectral flux

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the spectral flux across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the spectral flux across the segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of spectral flux
                        between consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of spectral
                       flux consecutive analysis points
        """

        flux = self._spectral_flux()

        mean, std, mean_diff, std_diff = self._feature_statistics_helper_two(flux)

        return mean, std, mean_diff, std_diff

    def _spectrum_spread_statistics(self):
        """
        compute statistical parameters of spectrum spread

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the spectrum spread across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the spectrum spread across the segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude of spectrum spread
                        between consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude of spectrum
                       spread consecutive analysis points
        """

        spread = self._spectrum_spread()

        mean, std, _, mean_diff, std_diff, __ = self._feature_statistics_helper_one(spread)

        return mean, std, mean_diff, std_diff

    def _LSTER(self):
        """
        compute the low short time energy ratio

        returns:
            LSTER : np.ndarray, shape=(n_segments,); the percentage of frames within the segment whose energy level
                    is below threshold of the average energy level across the segment
        """
        ste = self._short_time_energy()

        ste_segmented = lrs.util.frame(ste, frame_length=self._nframes_asegment, hop_length=self._nframes_asegment_hop).transpose()

        # reshape the mean can be broadcast
        ste_mean_asegment = np.mean(ste_segmented, axis=-1).reshape(-1,1)

        # this looks like more complex than we thought, because the output of np.sign function can be -1, 0 or 1.
        LSTER = 1.0 * np.sum((np.sign(np.sign(1.0/3 * ste_mean_asegment - ste_segmented) * 2 - 1) + 1), axis=-1) / (2 * self._nframes_asegment)

        return LSTER

    def _feature_statistics_concate(self):
        """
        compute all the feature statistics and concate them together

        returns:
            feature_statistics : np.ndarray, shape=(n_segments, n_feature_statistics); aaccroding to the index of columns,
                                 they are ['ste_mean', 'ste_std', 'ste_mean_diff', 'ste_std_diff',
                                           'zcr_mean', 'zcr_std', 'zcr_skewness', 'zcr_mean_diff', 'zcr_std_diff', 'zcr_skewness_diff',
                                           'lber_mean', 'lber_std', 'lber_mean_diff', 'lber_std_diff',
                                           'hber_mean', 'hber_std', 'hber_mean_diff', 'hber_std_diff',
                                           'corr_mean', 'corr_std', 'corr_mean_diff', 'corr_std_diff',
                                           'mfcc_mean', 'mfcc_std', 'mfcc_mean_diff', 'mfcc_std_diff',
                                           'mfccdn_mean', 'mfccdn_std', 'mfccdn_mean_diff', 'mfccdn_std_diff',
                                           'rolloff_mean', 'rolloff_std', 'rolloff_mean_diff', 'rolloff_std_diff',
                                           'centroid_mean', 'centroid_std', 'centroid_mean_diff', 'centroid_std_diff',
                                           'flux_mean', 'flux_std', 'flux_mean_diff', 'flux_std_diff',
                                           'spread_mean', 'spread_std', 'spread_mean_diff', 'spread_std_diff',
                                           'lster']
        """
        feature_statistics = None

        mean, std, mean_diff, std_diff = self._short_time_energy_statistics()

        # concate short time energy statistics
        # reshape to be concatable
        feature_statistics = mean.reshape(-1,1)
        feature_statistics = np.append(feature_statistics, std.reshape(-1,1), axis=-1)
        feature_statistics = np.append(feature_statistics, mean_diff.reshape(-1,1), axis=-1)
        feature_statistics = np.append(feature_statistics, std_diff.reshape(-1,1), axis=-1)
        
        # concate zero crossing rate ststistics
        # reshape to be concatalbe
        for statistic in self._zero_crossing_rate_statistics():
            feature_statistics = np.append(feature_statistics, statistic.reshape(-1,1), axis=-1)

        # concate low and high band energy ratio statistics
        # reshape to be concatable
        for statistic in self._band_energy_ratio_statistics():
            feature_statistics = np.append(feature_statistics, statistic.reshape(-1,1), axis=-1)

        # concate autocorralation coefficients statistics
        # reshape to be concate
        for statistic in self._autocorrelation_coefficient_statistics():
            feature_statistics = np.append(feature_statistics, statistic.reshape(-1,1), axis=-1)

        # concate mfcc statistics, it's different from other features, mfcc has self._mfcc_order columns, each column is one mfc
        for statistic in self._mfcc_statistics():
            feature_statistics = np.append(feature_statistics, statistic, axis=-1)

        # concate spectrum rolloff statistics
        # reshape to be concatable
        for statistic in self._spectrum_rolloff_statistics():
            feature_statistics = np.append(feature_statistics, statistic.reshape(-1,1), axis=-1)

        # concate spectrum centroid ststistics
        # reshape to be concatable
        for statistic in self._spectrum_centroid_statistics():
            feature_statistics = np.append(feature_statistics, statistic.reshape(-1,1), axis=-1)

        # concate spectrum flux ststistics
        # reshape to be concatable
        for statistic in self._spectral_flux_statistics():
            feature_statistics = np.append(feature_statistics, statistic.reshape(-1,1), axis=-1)

        # concate spectrum spread ststistics
        # reshape to be concatable
        for statistic in self._spectrum_spread_statistics():
            feature_statistics = np.append(feature_statistics, statistic.reshape(-1,1), axis=-1)

        # concate low short time energy ratio
        # reshape to be concatable
        feature_statistics = np.append(feature_statistics, self._LSTER().reshape(-1,1), axis=-1)

        return feature_statistics

    def _get_all_statistics(self):

        self._logger.info("Start Extracting all the training data's featute statistics!")

        # statistics inition; remember delete first row after the whole extraction
        speech_feature_statistics = np.ones(len(self._statistics_column_values)).reshape(1,-1)

        i = 1
        for root, dirlist, filelist in os.walk(self._speech_files_dir):
            for filename in filelist:
                if filename.split('.')[-1] == 'wav':

                    # load data, automatically resample the data to operating rate, and convert audio signal to mono if it's stero. 
                    src = os.path.join(root, filename)
                    lab = src.replace('.wav', '.lab')

                    self._data = self._load(src, lab, 'speech', sr=self._operating_rate, mono=True, dtype=np.float64)

                    if self._data is None:
                        self._logger.error("Load the speech data of {} failed!".format(src))
                        continue
                    else:
                        try:
                            speech_feature_statistics = np.append(speech_feature_statistics, self._feature_statistics_concate(), axis=0)
                        except:
                            self._logger.error("Compute the feature statistics of the speech data of {} failed!".format(src))
                            continue
                        else:
                            self._logger.info("{}'th speech file done!".format(i))
                            i = i + 1

        # delete the first row
        speech_feature_statistics = np.delete(speech_feature_statistics, 0, axis=0)

        # transform it to pandas.DataFrame and write it to disk
        speech_feature_statistics = pd.DataFrame(speech_feature_statistics, columns=self._statistics_column_values)

        self._logger.info("Extract the all speech data's featute statistics done!")


        # statistics inition; remember delete first row after the whole extraction
        music_feature_statistics = np.ones(len(self._statistics_column_values)).reshape(1,-1)

        i = 1
        for root, dirlist, filelist in os.walk(self._music_files_dir):
            for filename in filelist:
                if filename.split('.')[-1] == 'wav':

                    # load data, automatically resample the data to operating rate, and convert audio signal to mono if it's stero. 
                    src = os.path.join(root, filename)
                    lab = src.replace('.wav', '.lab')

                    self._data = self._load(src, lab, 'music', sr=self._operating_rate, mono=True, dtype=np.float64)

                    if self._data is None:
                        self._logger.error("Load the music data of {} failed!".format(src))
                        continue
                    else:
                        try:
                            music_feature_statistics = np.append(music_feature_statistics, self._feature_statistics_concate(), axis=0)
                        except:
                            self._logger.error("Compute the feature statistics of the music data of {} failed!".format(src))
                            continue
                        else:
                            self._logger.info("{}'th music file done!".format(i))
                            i = i + 1

        # delete the first row
        music_feature_statistics = np.delete(music_feature_statistics, 0, axis=0)

        # transform it to pandas.DataFrame and write it to disk
        music_feature_statistics = pd.DataFrame(music_feature_statistics, columns=self._statistics_column_values)

        self._logger.info("Extract the all music data's featute statistics done!")

        return speech_feature_statistics, music_feature_statistics

    def _write_all_statistics(self):
        """
        this function is used to extract all feature statistics to train
        """

        # extract all feature statistics for speech files
        speech_statistics_path = os.path.join(self._statistics_files_dir, self._speech_statistics_name)

        # extract all feature statistics for music files
        music_statistics_path = os.path.join(self._statistics_files_dir, self._music_statistics_name)

        speech_feature_statistic, music_feature_statistics = self._get_all_statistics()

        speech_feature_statistic.to_csv(speech_statistics_path+'.csv', index=False)
        self._logger.info("Write the all speech data's featute statistics done!")
        
        music_feature_statistics.to_csv(music_statistics_path+'.csv', index=False)
        self._logger.info("Write the all music data's featute statistics done!")

        return

    def _thresholds(self):
        pass

    def _write_feature_ranking(self):
        """
        get the feature ranking resault and write it to disk.
        """
        
        es_powerlist_ranking, em_powerlist_ranking, hs_powerlist_ranking, hm_powerlist_ranking, sp_powerlist_ranking =\
        self._feature_ranking()
        self._logger.info("writing feature ranking list to model files directory!")

        with open(os.path.join(self._model_files_dir, 'es_powerlist_ranking'), 'wb') as f:
            pickle.dump(es_powerlist_ranking, f)

        with open(os.path.join(self._model_files_dir, 'em_powerlist_ranking'), 'wb') as f:
            pickle.dump(em_powerlist_ranking, f)

        with open(os.path.join(self._model_files_dir, 'hs_powerlist_ranking'), 'wb') as f:
            pickle.dump(hs_powerlist_ranking, f)

        with open(os.path.join(self._model_files_dir, 'hm_powerlist_ranking'), 'wb') as f:
            pickle.dump(hm_powerlist_ranking, f)

        with open(os.path.join(self._model_files_dir, 'sp_powerlist_ranking'), 'wb') as f:
            pickle.dump(sp_powerlist_ranking, f)

        self._logger.info("Write feature ranking list to model files directory done!")

        return

    def _feature_ranking(self):
        """
        rank featuers
        return:
            es_powerlist_ranking : list of instances of Power, ranking extreme speech threshold separation power
            em_powerlist_ranking : list of instances of Power, ranking extreme music threshold separation power
            hs_powerlist_ranking : list of instances of Power, ranking high probability speech threshold separation power
            hm_powerlist_ranking : list of instances of Power, ranking high probability music threshold separation power
            sp_powerlist_ranking : list of instances of Power, ranking separation threshold separation power
        """
        logging.info("loading the speech and music statistics dataframe!")
        # extract all feature statistics for speech files
        speech_statistics_path = os.path.join(self._statistics_files_dir, self._speech_statistics_name+'.csv')

        # extract all feature statistics for music files
        music_statistics_path = os.path.join(self._statistics_files_dir, self._music_statistics_name+'.csv')

        # load speech and music statistics data
        speech = pd.read_csv(speech_statistics_path)
        music = pd.read_csv(music_statistics_path)
        data = pd.concat([speech, music], ignore_index=True)

        # compute separetion power for every feature,
        power_list = []

        self._logger.info("Starting computing threshold separation power!")
        for name in self._statistics_column_values:
            self._logger.info("Computing threshold separation power for feature: {}!".format(name))
            power_list.append(Power(name, speech[name].values, music[name].values, self._logger))
        self._logger.info("Compute threshold separation power done!")

        # save power list
        with open(os.path.join(self._model_files_dir, 'power_list'), 'wb') as f:
            pickle.dump(power_list, f)

        self._logger.info("Computing cos simility for every pair of feature statistics")
        # compute cos simility for every pair of feature statistics
        cos_similities = {}
        for i in range(len(self._statistics_column_values)):
            for j in range(i+1, len(self._statistics_column_values)):
                cos_simility = \
                abs((data[self._statistics_column_values[i]].values * data[self._statistics_column_values[j]].values).sum() / \
                (np.sqrt((data[self._statistics_column_values[i]].values**2).sum()) * np.sqrt((data[self._statistics_column_values[j]].values**2).sum())))
                
                cos_similities[self._statistics_column_values[i] + '_' + self._statistics_column_values[j]] = cos_simility
                
                cos_similities[self._statistics_column_values[j] + '_' + self._statistics_column_values[i]] = cos_simility
                
        self._logger.info("Computing cos simility for every pair of feature statistics")

        # ********************************************************************************************************
        # **************************compute extreme speech threshold features' ranking****************************
        # ********************************************************************************************************

        self._logger.info("Computing extreme speech threshold separation power features' ranking")
        # remaining features
        powerlist_remaining = copy.deepcopy(power_list)

        # ranked features
        es_powerlist_ranking = []

        # initialize highest score and name of the feature who has the highest score for now
        highest_score = -np.inf
        name_selected = ''

        # find the feature which has highest extreme speech separetion power
        for power_remaining in powerlist_remaining:
            if power_remaining.es_power_ > highest_score:
                highest_score = power_remaining.es_power_
                name_selected = power_remaining.name_

        # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to es_powerlist_ranking
        for power_remaining in powerlist_remaining:
            if power_remaining.name_ == name_selected:
                es_powerlist_ranking.append(power_remaining)
                powerlist_remaining.remove(power_remaining)

        while len(powerlist_remaining) != 0:
            # initialize highest score and name of the feature who has the highest score for now
            highest_score = -np.inf
            name_selected = ''

            for power_remaining in powerlist_remaining:
                # initialize the sum of mutual correlation between power(feature) remaining and each power(feature) ranked
                corr_sum = 0
                
                for power_ranking in es_powerlist_ranking:
                    corr_sum = corr_sum + cos_similities[power_remaining.name_+'_'+power_ranking.name_]
                
                score = self._alpha * power_remaining.es_power_ - self._beta * corr_sum / len(es_powerlist_ranking)
                
                # if the usefullness score of this feature is higher than the highest_score, change the highest_score and name_selected
                if score > highest_score:
                    highest_score = score
                    name_selected = power_remaining.name_

            # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to es_powerlist_ranking
            for power_remaining in powerlist_remaining:
                if power_remaining.name_ == name_selected:
                    es_powerlist_ranking.append(power_remaining)
                    powerlist_remaining.remove(power_remaining)

        self._logger.info("Compute extreme speech threshold separation power features' ranking done!")

        # ********************************************************************************************************
        # **************************compute extreme music threshold features' ranking****************************
        # ********************************************************************************************************
        self._logger.info("Computing extreme music threshold separation power features' ranking")
        # remaining features
        powerlist_remaining = copy.deepcopy(power_list)

        # ranked features
        em_powerlist_ranking = []

        # initialize highest score and name of the feature who has the highest score for now
        highest_score = -np.inf
        name_selected = ''

        # find the feature which has highest extreme music separetion power
        for power_remaining in powerlist_remaining:
            if power_remaining.em_power_ > highest_score:
                highest_score = power_remaining.em_power_
                name_selected = power_remaining.name_

        # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to em_powerlist_ranking
        for power_remaining in powerlist_remaining:
            if power_remaining.name_ == name_selected:
                em_powerlist_ranking.append(power_remaining)
                powerlist_remaining.remove(power_remaining)

        while len(powerlist_remaining) != 0:
            # initialize highest score and name of the feature who has the highest score for now
            highest_score = -np.inf
            name_selected = ''

            for power_remaining in powerlist_remaining:
                # initialize the sum of mutual correlation between power(feature) remaining and each power(feature) ranked
                corr_sum = 0
                
                for power_ranking in em_powerlist_ranking:
                    corr_sum = corr_sum + cos_similities[power_remaining.name_+'_'+power_ranking.name_]
                
                score = self._alpha * power_remaining.em_power_ - self._beta * corr_sum / len(em_powerlist_ranking)
                
                # if the usefullness score of this feature is higher than the highest_score, change the highest_score and name_selected
                if score > highest_score:
                    highest_score = score
                    name_selected = power_remaining.name_

            # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to es_powerlist_ranking
            for power_remaining in powerlist_remaining:
                if power_remaining.name_ == name_selected:
                    em_powerlist_ranking.append(power_remaining)
                    powerlist_remaining.remove(power_remaining)

        self._logger.info("Compute extreme music threshold separation power features' ranking done!")

        # ********************************************************************************************************
        # **************************compute high probability speech threshold features' ranking****************************
        # ********************************************************************************************************
        self._logger.info("Computing high probability speech threshold separation power features' ranking")
        # remaining features
        powerlist_remaining = copy.deepcopy(power_list)

        # ranked features
        hs_powerlist_ranking = []

        # initialize highest score and name of the feature who has the highest score for now
        highest_score = -np.inf
        name_selected = ''

        # find the feature which has highest high probability speech separetion power
        for power_remaining in powerlist_remaining:
            if power_remaining.hs_power_ > highest_score:
                highest_score = power_remaining.hs_power_
                name_selected = power_remaining.name_

        # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to em_powerlist_ranking
        for power_remaining in powerlist_remaining:
            if power_remaining.name_ == name_selected:
                hs_powerlist_ranking.append(power_remaining)
                powerlist_remaining.remove(power_remaining)

        while len(powerlist_remaining) != 0:
            # initialize highest score and name of the feature who has the highest score for now
            highest_score = -np.inf
            name_selected = ''

            for power_remaining in powerlist_remaining:
                # initialize the sum of mutual correlation between power(feature) remaining and each power(feature) ranked
                corr_sum = 0
                
                for power_ranking in hs_powerlist_ranking:
                    corr_sum = corr_sum + cos_similities[power_remaining.name_+'_'+power_ranking.name_]
                
                score = self._alpha * power_remaining.hs_power_ - self._beta * corr_sum / len(hs_powerlist_ranking)
                
                # if the usefullness score of this feature is higher than the highest_score, change the highest_score and name_selected
                if score > highest_score:
                    highest_score = score
                    name_selected = power_remaining.name_

            # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to es_powerlist_ranking
            for power_remaining in powerlist_remaining:
                if power_remaining.name_ == name_selected:
                    hs_powerlist_ranking.append(power_remaining)
                    powerlist_remaining.remove(power_remaining)

        self._logger.info("Compute high probability speech threshold separation power features' ranking done!")
        # ********************************************************************************************************
        # **************************compute high probability music threshold features' ranking****************************
        # ********************************************************************************************************
        self._logger.info("Computing high probability music threshold separation power features' ranking")
        # remaining features
        powerlist_remaining = copy.deepcopy(power_list)

        # ranked features
        hm_powerlist_ranking = []

        # initialize highest score and name of the feature who has the highest score for now
        highest_score = -np.inf
        name_selected = ''

        # find the feature which has highest high probability speech separetion power
        for power_remaining in powerlist_remaining:
            if power_remaining.hm_power_ > highest_score:
                highest_score = power_remaining.hm_power_
                name_selected = power_remaining.name_

        # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to em_powerlist_ranking
        for power_remaining in powerlist_remaining:
            if power_remaining.name_ == name_selected:
                hm_powerlist_ranking.append(power_remaining)
                powerlist_remaining.remove(power_remaining)

        while len(powerlist_remaining) != 0:
            # initialize highest score and name of the feature who has the highest score for now
            highest_score = -np.inf
            name_selected = ''

            for power_remaining in powerlist_remaining:
                # initialize the sum of mutual correlation between power(feature) remaining and each power(feature) ranked
                corr_sum = 0
                
                for power_ranking in hm_powerlist_ranking:
                    corr_sum = corr_sum + cos_similities[power_remaining.name_+'_'+power_ranking.name_]
                
                score = self._alpha * power_remaining.hm_power_ - self._beta * corr_sum / len(hm_powerlist_ranking)
                
                # if the usefullness score of this feature is higher than the highest_score, change the highest_score and name_selected
                if score > highest_score:
                    highest_score = score
                    name_selected = power_remaining.name_

            # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to es_powerlist_ranking
            for power_remaining in powerlist_remaining:
                if power_remaining.name_ == name_selected:
                    hm_powerlist_ranking.append(power_remaining)
                    powerlist_remaining.remove(power_remaining)

        self._logger.info("Compute high probability music threshold separation power features' ranking done!")

        # ********************************************************************************************************
        # **************************compute separation threshold features' ranking****************************
        # ********************************************************************************************************
        self._logger.info("Computing separation threshold separation power features' ranking")
        # remaining features
        powerlist_remaining = copy.deepcopy(power_list)

        # ranked features
        sp_powerlist_ranking = []

        # initialize highest score and name of the feature who has the highest score for now
        highest_score = -np.inf
        name_selected = ''

        # find the feature which has highest separation threshold separetion power
        for power_remaining in powerlist_remaining:
            if power_remaining.sp_power_ > highest_score:
                highest_score = power_remaining.sp_power_
                name_selected = power_remaining.name_

        # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to em_powerlist_ranking
        for power_remaining in powerlist_remaining:
            if power_remaining.name_ == name_selected:
                sp_powerlist_ranking.append(power_remaining)
                powerlist_remaining.remove(power_remaining)

        while len(powerlist_remaining) != 0:
            # initialize highest score and name of the feature who has the highest score for now
            highest_score = -np.inf
            name_selected = ''

            for power_remaining in powerlist_remaining:
                # initialize the sum of mutual correlation between power(feature) remaining and each power(feature) ranked
                corr_sum = 0
                
                for power_ranking in sp_powerlist_ranking:
                    corr_sum = corr_sum + cos_similities[power_remaining.name_+'_'+power_ranking.name_]
                
                score = self._alpha * power_remaining.sp_power_ - self._beta * corr_sum / len(sp_powerlist_ranking)
                
                # if the usefullness score of this feature is higher than the highest_score, change the highest_score and name_selected
                if score > highest_score:
                    highest_score = score
                    name_selected = power_remaining.name_

            # search the power whose name is name_selected in powerlist_remaining, remove it from powerlist_remaining and append it back to es_powerlist_ranking
            for power_remaining in powerlist_remaining:
                if power_remaining.name_ == name_selected:
                    sp_powerlist_ranking.append(power_remaining)
                    powerlist_remaining.remove(power_remaining)
        self._logger.info("Compute separation threshold separation power features' ranking done!")

        return es_powerlist_ranking, em_powerlist_ranking, hs_powerlist_ranking, hm_powerlist_ranking, sp_powerlist_ranking

    def _cross_validation(self):
        """
        """
        self._logger.info("Start cross validation to choose features!")
        K_fold = 5
        # maximum features for each threshold
        max_feature = 15
        # dict of cv score
        cv_score = {}

        self._logger.info("loading the speech and music statistics dataframe!")
        # extract all feature statistics for speech files
        speech_statistics_path = os.path.join(self._statistics_files_dir, self._speech_statistics_name+'.csv')

        # extract all feature statistics for music files
        music_statistics_path = os.path.join(self._statistics_files_dir, self._music_statistics_name+'.csv')

        # load speech and music statistics data
        speech = pd.read_csv(speech_statistics_path)
        music = pd.read_csv(music_statistics_path)

        self._logger.info("loading all kinds of ranking list!")
        # load all kinds of ranking list
        with open(os.path.join(self._model_files_dir, 'es_powerlist_ranking'), 'r') as f:
            es_powerlist_ranking = pickle.load(f)
            es_features = [power.name_ for power in es_powerlist_ranking[0:max_feature]]

        with open(os.path.join(self._model_files_dir, 'em_powerlist_ranking'), 'r') as f:
            em_powerlist_ranking = pickle.load(f)
            em_features = [power.name_ for power in em_powerlist_ranking[0:max_feature]]

        with open(os.path.join(self._model_files_dir, 'hs_powerlist_ranking'), 'r') as f:
            hs_powerlist_ranking = pickle.load(f)
            hs_features = [power.name_ for power in hs_powerlist_ranking[0:max_feature]]
            hs_positions = np.array([-1 if power._speech_position == 'left' else 1 for power in hs_powerlist_ranking[0:max_feature]])

        with open(os.path.join(self._model_files_dir, 'hm_powerlist_ranking'), 'r') as f:
            hm_powerlist_ranking = pickle.load(f)
            hm_features = [power.name_ for power in hm_powerlist_ranking[0:max_feature]]
            hm_positions = np.array([-1 if power._music_position == 'left' else 1 for power in hm_powerlist_ranking[0:max_feature]])

        with open(os.path.join(self._model_files_dir, 'sp_powerlist_ranking'), 'r') as f:
            sp_powerlist_ranking = pickle.load(f)
            sp_features = [power.name_ for power in sp_powerlist_ranking[0:max_feature]]
            sp_music_positions = np.array([-1 if power._music_position == 'left' else 1 for power in sp_powerlist_ranking[0:max_feature]])
            sp_speech_positions = -1 * sp_music_positions

        # start cross-validation
        rs = ShuffleSplit(n_splits=K_fold, test_size=0.2, train_size=0.8, random_state=2018)
        # list of speech/music train/test index list
        speech_train_ix_list = []
        speech_test_ix_list = []
        music_train_ix_list = []
        music_test_ix_list = []

        # genarate list of speech/music train/test index list
        for speech_train_ix, speech_test_ix in rs.split(speech.values):
            speech_train_ix_list.append(speech_train_ix)
            speech_test_ix_list.append(speech_test_ix)
        for music_train_ix, music_test_ix in rs.split(music.values):
            music_train_ix_list.append(music_train_ix)
            music_test_ix_list.append(music_test_ix)

        for i in range(K_fold):
            # fetch speech/music train/test dataframe using index operation
            speech_train = speech.iloc[speech_train_ix_list[i]]
            speech_test = speech.iloc[speech_test_ix_list[i]]
            music_train = music.iloc[music_train_ix_list[i]]
            music_test = music.iloc[music_test_ix_list[i]]

            # copy the ranking list for each threshold and update corresponding threshold using cv train data
            es_powerlist_ranking_cv = copy.deepcopy(es_powerlist_ranking[0:max_feature])
            em_powerlist_ranking_cv = copy.deepcopy(em_powerlist_ranking[0:max_feature])
            hs_powerlist_ranking_cv = copy.deepcopy(hs_powerlist_ranking[0:max_feature])
            hm_powerlist_ranking_cv = copy.deepcopy(hm_powerlist_ranking[0:max_feature])
            sp_powerlist_ranking_cv = copy.deepcopy(sp_powerlist_ranking[0:max_feature])
            
            # update extreme speech threshold using speech/music training data
            for power in es_powerlist_ranking_cv:
                power.update_extreme_threshold(speech_train[power.name_].values, music_train[power.name_].values)
            
            # fetch the new extrame speech thresholds
            es_thresholds_left = np.array([power.extreme_speech_left_ for power in es_powerlist_ranking_cv])
            es_thresholds_right = np.array([power.extreme_speech_right_ for power in es_powerlist_ranking_cv])
            
            # update extreme music threshold using speech/music training data
            for power in em_powerlist_ranking_cv:
                power.update_extreme_threshold(speech_train[power.name_].values, music_train[power.name_].values)
            
            # fetch the new extrame music thresholds
            em_thresholds_left = np.array([power.extreme_music_left_ for power in em_powerlist_ranking_cv])
            em_thresholds_right = np.array([power.extreme_music_right_ for power in em_powerlist_ranking_cv])
            
            # update high probability speech threshold using speech/music training data
            for power in hs_powerlist_ranking_cv:
                power.update_high_probability_threshold(speech_train[power.name_].values, music_train[power.name_].values)
            
            # fetch the new high probability speech thresholds
            hs_thresholds = np.array([power.high_speech_ for power in hs_powerlist_ranking_cv])

            # update high probability music threshold using speech/music training data
            for power in hm_powerlist_ranking_cv:
                power.update_high_probability_threshold(speech_train[power.name_].values, music_train[power.name_].values)
            
            # fetch the new high probability speech thresholds
            hm_thresholds = np.array([power.high_music_ for power in hm_powerlist_ranking_cv])

            # update separation thresholds using speech/music training data
            for power in sp_powerlist_ranking_cv:
                power.update_separation_threshold(speech_train[power.name_].values, music_train[power.name_].values)
            
            # fetch new separation thresholds
            sp_thresholds = np.array([power.separation_ for power in sp_powerlist_ranking_cv])


            def segmentation(feature_statistics_df, a, b, c, d, e):
                """
                perform segmentation of a given audio signal's feature statistics dataframe into “speech” and “music”.

                parameters:
                    feature_statistics_df : pandas.DataFrame, shape=(n_segments, m_features)
                    a : int, number of features used in extreme speech threshold
                    b : int, number of features used in extreme music threshold
                    c : int, number of features used in high probability speech threshold
                    d : int, number of features used in high probability music threshold
                    e : int, number of features used in separation threshold
                
                return:
                    Di : np.ndarray, shape=(n_segments,), -1 or 1 or in (-1,1), -1 -> music, 1 -> speech
                """
                # initialize the alpha, mentioned at page 9 of the paper
                alpha = 0.66666

                # number of features above its corresponding extrem speech threshold
                S_ex_left = np.where((feature_statistics_df[es_features[0:a]].values - es_thresholds_left[0:a]) < 0, 1, 0)
                S_ex_right = np.where((feature_statistics_df[es_features[0:a]].values - es_thresholds_right[0:a]) > 0, 1, 0)
                S_x = np.sum(S_ex_left + S_ex_right, axis=-1)

                # number of features above its corresponding high probability speech threshold
                S_h = np.sum(np.where(((feature_statistics_df[hs_features[0:c]].values - hs_thresholds[0:c]) * hs_positions[0:c]) > 0, 1, 0) , axis=-1)

                # number of features in the separation set that are classified as speech
                S_p = np.sum(np.where(((feature_statistics_df[sp_features[0:e]].values - sp_thresholds[0:e]) * sp_speech_positions[0:e]) > 0, 1, 0) , axis=-1)

                # number of features above its corresponding extrem music threshold
                M_ex_left = np.where((feature_statistics_df[em_features[0:b]].values - em_thresholds_left[0:b]) < 0, 1, 0)
                M_ex_right = np.where((feature_statistics_df[em_features[0:b]].values - em_thresholds_right[0:b]) > 0, 1, 0)
                M_x = np.sum(M_ex_left + M_ex_right, axis=-1)

                # number of features above its corresponding high probability music threshold
                M_h = np.sum(np.where(((feature_statistics_df[hm_features[0:d]].values - hm_thresholds[0:d]) * hm_positions[0:d]) > 0, 1, 0) , axis=-1)

                 # number of features in the separation set that are classified as music
                M_p = np.sum(np.where(((feature_statistics_df[sp_features[0:e]].values - sp_thresholds[0:e]) * sp_music_positions[0:e]) > 0, 1, 0) , axis=-1)

                # number of segments
                n_segments = feature_statistics_df.count()[0]

                # initial classification label
                Di = np.zeros(n_segments)

                # initial classification for each segment
                for j in range(n_segments):
                    if (S_x > 0 and M_x == 0 and M_h == 0) or (S_x > 1 and Mx == 0) or (S_h > alpha*c and M_h == 0):
                        Di[j] = 1.0
                    elif (M_x > 0 and S_x == 0 and S_x == 0) or (M_x > 1 and S_x == 0) or (M_h > alpha*d and S_h == 0):
                        Di[j] -1.0
                    else:
                        Di[j] = 1.0 * (S_p - M_p) / e

                return Di

            # cross-validation to get scores
            for a in range(1, max_feature+1, 2): # a is number of features used in extreme speech threshold
                for b in range(1, max_feature+1, 2): # b is number of features used in extreme music threshold
                    for c in range(1, max_feature+1, 2): # c is number of features used in high probability speech threshold
                        for d in range(1, max_feature+1, 2): # d is number of features used in high probability music threshold
                            for e in range(1, max_feature+1, 2): # e is number of features used in separation threshold
                                speech_test_resault = segmentation(speech_test,a,b,c,d,e)
                                music_test_resault = segmentation(music_test,a,b,c,d,e)
                                test_score = np.where([speech_test_resault>0,music_test_resault<0], 1, 0).mean()
                                if cv_score.get('{}_{}_{}_{}_{}'.format(a,b,c,d,e), None) == None:
                                    cv_score['{}_{}_{}_{}_{}'.format(a,b,c,d,e)] = test_score
                                else:
                                    cv_score['{}_{}_{}_{}_{}'.format(a,b,c,d,e)] += test_score

                                self._logger.info("{}'th cross-validation fold!\n extreme speech features number: {}\n extreme music features number: {}\n, high speech features number: {}\n, high music features number: {}, separation features number: {}\n, accuracy: {}".format(i+1,a,b,c,d,e,test_score))

            














    def _feature_statistics_helper_one(self, feature, skew=False):
        """
        help compute the statistical parameters of one feature.
        
        parameters:
            feature : np.ndarray, shape=(n_frames,); one kind of feature for each framed audio
            skew : bool; if true, compute the skewness of feature and skewness of the difference between
                   consecutive analysis frames; default is False

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the feature across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the feature across the segment
            skewness : np.ndarray, shape=(n_segments,); skewness of the feature in one segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude between
                        consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude
                       between consecutive analysis points
            skewness_diff : skewness of the difference between consecutive analysis frames
        
        notes:
            this function is only used to help compute ste, zcr, low(high) band energy ratio,
            autocorrelation coefficients, mfcc(not mfcc diff norm), spectrum roll-off, spect-
            rum centroid and spectum spread statistical parameters. The statistical parameters
            of mfcc diff norm and spectral flux can not use this function to compute.
            Please use helper two.
        """
        skewness = None
        skewness_diff = None

        feature_segmented = lrs.util.frame(feature, frame_length=self._nframes_asegment, hop_length=self._nframes_asegment_hop).transpose()

        mean = np.mean(feature_segmented, axis=-1)

        std = np.std(feature_segmented, axis=-1)

        feature_diff = feature[1:] - feature[0:-1]

        feature_diff_segmented = lrs.util.frame(feature_diff, frame_length=self._nframes_asegment-1, hop_length=self._nframes_asegment_hop).transpose()

        mean_diff = np.mean(feature_diff_segmented, axis=-1)

        std_diff = np.std(feature_diff_segmented, axis=-1)

        if skew:
            skewness = np.mean((feature_segmented-mean.reshape(-1,1))**3, axis=-1) / (std**3+0.0000001)
            skewness_diff = np.mean((feature_diff_segmented-mean_diff.reshape(-1,1))**3, axis=-1) / (std_diff**3 + 0.0000001)

        return mean, std, skewness, mean_diff, std_diff, skewness_diff

    def _feature_statistics_helper_two(self, feature):
        """
        help compute the statistical parameters of features: mfcc diff norm, spectral flux
        
        parameters:
            feature : np.ndarray, shape=(n_frames-1,)

        returns:
            mean : np.ndarray, shape=(n_segments,); mean value of the feature across the segment
            std : np.ndarray, shape=(n_segments,); standard deviation of the feature across the segment
            mean_diff : np.ndarray, shape=(n_segments,); mean value of the difference magnitude between
                        consecutive analysis points
            std_diff : np.ndarray, shape=(n_segments,); standard deviation of the difference magnitude
                       between consecutive analysis points

        notes:
            this function is only used to help compute mfcc diff norm, spectral flux statistical parameters.
            See helper one.
        """

        feature_segmented = lrs.util.frame(feature, frame_length=self._nframes_asegment-1, hop_length=self._nframes_asegment_hop).transpose()

        mean = np.mean(feature_segmented, axis=-1)

        std = np.std(feature_segmented, axis=-1)

        feature_diff = feature[1:] - feature[0:-1]

        feature_diff_segmented = lrs.util.frame(feature_diff, frame_length=self._nframes_asegment-2, hop_length=self._nframes_asegment_hop).transpose()

        mean_diff = np.mean(feature_diff_segmented, axis=-1)

        std_diff = np.std(feature_diff_segmented, axis=-1)

        return mean, std, mean_diff, std_diff

    def _stft(self):
        """
        Short-time Fourier transform (STFT)

        Returns a real matrix stft_matrix such that
            stft_matrix[t, f] is the magnitude of frequency bin `f`
            at frame `t`

            stft_matrix[t, f] is the phase of frequency bin `f`
            at frame `t`

        Returns
        -------
        stft_matrix : np.ndarray [shape=(t, 1 + n_fft/2), dtype=np.float64?]
        """

        fft_window = lrs.filters.get_window('hann', self._n_fft, fftbins=True)

        # Reshape so that the window can be broadcast
        fft_window = fft_window.reshape((-1, 1))

        # Window the time series.
        y_frames = lrs.util.frame(self._data, frame_length=self._frame_length, hop_length=self._frame_hop_length)

        # pad the framed data out to n_fft size
        y_frames = lrs.util.pad_center(y_frames, self._n_fft, axis=0)

        # Pre-allocate the STFT matrix
        stft_matrix = np.empty((int(1 + self._n_fft // 2), y_frames.shape[1]),
                               dtype=np.complex64,
                               order='F')

        # how many columns can we fit within MAX_MEM_BLOCK?
        n_columns = int(lrs.util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                              stft_matrix.itemsize))

        for bl_s in range(0, stft_matrix.shape[1], n_columns):
            bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

            # `np.abs(stft_matrix[f, t])` is the magnitude of frequency bin `f`
            # at frame `t`

            # `np.angle(stft_matrix[f, t])` is the phase of frequency bin `f`
            # at frame `t`
            # RFFT and Conjugate here to match phase from DPWE code
            stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                                y_frames[:, bl_s:bl_t],
                                                axis=0)[:stft_matrix.shape[0]]

        # get the magnitude of the fft matrix and transpose it to make axis=0 to indicate frame index.
        stft_matrix = np.abs(stft_matrix.transpose())

        return stft_matrix

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hc:',['help', 'config='])
    except getopt.GetoptError as e:
        print('python Learn.py -c <config file path>')

    for opt, value in opts:
        if opt in ['-h', '--help']:
            print('python Learn.py -c <config file path>')
        elif opt in ['-c', '--config']:
            cfg = value
    try:
        Train(cfg)
    except UnboundLocalError:
        print('python Learn.py -c <config file path>')

if __name__ == '__main__':
    main(sys.argv[1:])