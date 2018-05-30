from Learn import Power

import zipfile
import yaml
import pickle
import os
import logging
import sys
import getopt

import pandas as pd
import numpy as np



class Test(object):
    """
    this class is to perform segmentaion of a given audio signal into speech and music.

    parameters:
        system : string; system setting zip file path
        cfg : string; config file path

    attributes:
        es_ranking_list : list of Power instances, ranked by extrem speech threshold separation power
        em_ranking_list : list of Power instances, ranked by extrem music threshold separation power
        hs_ranking_list : list of Power instances, ranked by high probability speech threshold separation power
        hm_ranking_list : list of Power instances, ranked by high probability music threshold separation power
        sp_ranking_list : list of Power instances, ranked by speech, music separation threshold separation power
        setting : dict, dict of system setting
        config : dict, dict of target configeration

        es_features :
        es_thresholds_left :
        es_thresholds_right :
        em_features :
        em_thresholds_left :
        em_thresholds_right :
        hs_features :
        hs_thresholds :
        hs_positions :
        hm_features :
        hm_thresholds :
        hm_positions :
        sp_features :
        sp_thresholds :
        sp_music_positions :
        sp_speech_positions :

        logger : logging.Logger

        _operating_rate :
        _frame_length :
        _frame_hop_length :
        _segment_length :
        _segment_hop_length :
        _n_fft :
        _mfcc_order :
        _roll_percent :
        _statistics_column_values :
        T_init :
        T_min :

    """
    def __init__(self, system, cfg):

        # ***************************system setting configuration******************************
        try:
            with zipfile.ZipFile(system) as myzip:
                # load extrem speech threshold separation power ranking list
                with myzip.open('es_ranking_list', 'r') as f:
                    self.es_ranking_list = pickle.load(f)

                # load extrem music threshold separation power ranking list
                with myzip.open('em_ranking_list', 'r') as f:
                    self.em_ranking_list = pickle.load(f)

                # load high probability speech threshold separation power ranking list
                with myzip.open('hs_ranking_list', 'r') as f:
                    self.hs_ranking_list = pickle.load(f)

                # load high probability music threshold separation power ranking list
                with myzip.open('hm_ranking_list', 'r') as f:
                    self.hm_ranking_list = pickle.load(f)

                # load speech, music separation threshold separation power ranking list
                with myzip.open('sp_ranking_list', 'r') as f:
                    self.sp_ranking_list = pickle.load(f)

                # load system setting dict
                with myzip.open('setting.yaml', 'r') as f:
                    self.setting = yaml.load(f)
        except:
            raise ValueError("system setting file is damaged, please check: {}".format(system))

        # system setting values
        self._operating_rate = self.setting['operating_rate']
        self._frame_length = int(self.setting['operating_rate'] * self.setting['frame_size'])
        self._frame_hop_length = int(self.setting['operating_rate'] * self.setting['frame_shift'])
        self._segment_length = int(self.setting['operating_rate'] * self.setting['segment_size'])
        self._segment_hop_length = int(self.setting['operating_rate'] * self.setting['segment_shift'])
        self._n_fft = self.setting['n_fft']
        self._mfcc_order = self.setting['mfcc_order']
        self._roll_percent = self.setting['roll_percent']

        # adaptive threshold setting according to the number of features used in separation threshold
        self.T_init = 1.0*4 / self.setting['n_sp_features'] - 0.01

        self.T_min = 1.0*2 / self.setting['n_sp_features'] - 0.01

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

        # the best n features for extreme speech threshold and its corresponding extrem speech threshold
        self.es_features = [power.name_ for power in self.es_ranking_list[0:self.setting['n_es_features']]]
        self.es_thresholds_left = np.array([power.extreme_speech_left_ for power in self.es_ranking_list[0:self.setting['n_es_features']]])
        self.es_thresholds_right = np.array([power.extreme_speech_right_ for power in self.es_ranking_list[0:self.setting['n_es_features']]])

        # the best n features for extreme music threshold and its corresponding extrem music threshold
        self.em_features = [power.name_ for power in self.em_ranking_list[0:self.setting['n_em_features']]]
        self.em_thresholds_left = np.array([power.extreme_music_left_ for power in self.em_ranking_list[0:self.setting['n_em_features']]])
        self.em_thresholds_right = np.array([power.extreme_music_right_ for power in self.em_ranking_list[0:self.setting['n_em_features']]])

        # the best n features for high probibality speech threshold and its corresponding high propability speech threshold,
        # speech/music statistics position, -1 indicate 'left', 1 indicate right
        self.hs_features = [power.name_ for power in self.hs_ranking_list[0:self.setting['n_hs_features']]]
        self.hs_thresholds = np.array([power.high_speech_ for power in self.hs_ranking_list[0:self.setting['n_hs_features']]])
        self.hs_positions = np.array([-1 if power._speech_position == 'left' else 1 for power in self.hs_ranking_list[0:self.setting['n_hs_features']]])

        # the best n features for high probibality music threshold and its corresponding high propability music threshold
        # speech/music position, -1 indicate 'left', 1 indicate right
        self.hm_features = [power.name_ for power in self.hm_ranking_list[0:self.setting['n_hm_features']]]
        self.hm_thresholds = np.array([power.hm_threshold_ for power in self.hm_ranking_list[0:self.setting['n_hm_features']]])
        self.hm_positions = np.array([-1 if power._music_position == 'left' else 1 for power in self.hm_ranking_list[0:selt.setting['n_hm_features']]])

        # the best n features for separation threshold and its corresponding separation threshold
        # speech/music position, -1 indicate 'left', 1 indicate right
        self.sp_features = [power.name_ for power in self.sp_ranking_list[0:self.setting['n_sp_features']]]
        self.sp_thresholds = np.array([power.separation_ for power in self.sp_ranking_list[0:self.setting['n_sp_features']]])
        self.sp_music_positions = np.array([-1 if power._music_position == 'left' else 1 for power in self.sp_ranking_list[0:self.setting['n_sp_features']]])
        self.sp_speech_positions = -1 * self.sp_music_positions


        # target config
        try:
            with open(cfg, 'r') as f:
                self.config = yaml.load(f)
        except:
            raise ValueError("Loading Target config file failed, please check: {}".format(cfg))

        # *****************************check target config*******************************
        # check log file path and initialize logger
        if self.config.get('log_file_path', None) == None:
            raise ValueError("Please set log file path in cfg file: {}".format(cfg))
        else:
            log_dir = os.path.split(self.config('log_file_path'))[0]
            if not os.path.exists(log_dir):
                raise ValueError("logging dir does not exist: {}".format(log_dir))
            else:
                # logger configuration
                logging.basicConfig(filename=self.config('log_file_path'),
                                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                    filemode='w',
                                    level=logging.DEBUG)
                self.logger = logging.getLogger('Testing procession')

        # tag with statistics files
        if self.config.get('with_statistics', None) == True:
            # if tag the file using features, then check the feature files directory
            if self.config.get('feature_path', None) == None:
                raise ValueError("Tag with statistics, but you don't give the statistics file path, please set statistics path!")
            if not os.path.exists(self.config['feature_path']):
                raise ValueError("Tag with statistics, but the statistics path you give doesn't exist: {}, please check!".format(self.config['feature_path']))
            
            # check labels file directory, if not given, set it same as feature files path, if given bu doesn't exist, create it
            if self.config.get('labels_path', None) == None:
                if os.path.isfile(self.config['feature_path']):
                    self.config['labels_path'] = os.path.basedir(self.config['feature_path'])
                else:
                    self.config['labels_path'] = self.config['feature_path']
            else:
                if not os.path.exists(self.config['labels_path']):
                    try:
                        os.makedirs(self.config['labels_path'])
                    except Exception as e:
                        raise(e)

        # tag with wav files
        else:
            # check source data files path
            if self.config.get('wav_path', None) == None:
                raise ValueError("Please give a correct data files path in cfg file: {}".format(cfg))

            if not os.path.exists(self.config['wav_path']):
                raise ValueError("Source data files path does not exist!: {}".format(elf.config['wav_path']))
            
            # check statistics files path, if not given, set it same as wave files path, if given bu doesn't exist, create it
            if self.config.get('feature_path', None) == None:
                if os.path.isfile(self.config['wav_path']):
                    self.config['feature_path'] = os.path.basedir(self.config['wav_path'])
                else:
                    self.config['feature_path'] = self.config['wav_path']
            else:
                if not os.path.exists(self.config['feature_path']):
                    try:
                        os.makedirs(self.config['feature_path'])
                    except Exception as e:
                        raise(e)

            # check labels file directory
            if self.config.get('labels_path', None) == None:
                if os.path.isfile(self.config['wav_path']):
                    self.config['labels_path'] = os.path.basedir(self.config['wav_path'])
                else:
                    self.config['labels_path'] = self.config['wav_path']
            else:
                if not os.path.exists(self.config['labels_path']):
                    try:
                        os.makedirs(self.config['labels_path'])
                    except Exception as e:
                        raise(e)

        # check smoothing parameters
        if self.config.get('average_period', None) == None:
            self.config['average_period'] = 5
        if self.config.get('time_constant', None) == None:
            self.config['time_constant'] = self.config['average_period']

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

    def _all_statistics_concate(self):
        """
        compute all the feature statistics for one wave file and concate them together

        returns:
            feature_statistics_df : pandas.DataFrame, shape=(n_segments, n_feature_statistics);
                                    columns.values are ['ste_mean', 'ste_std', 'ste_mean_diff', 'ste_std_diff',
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

        # pd.dataFrame
        feature_statistics_df = pd.DataFrame(feature_statistics, columns=self._statistics_column_values)

        return feature_statistics_df

    def 



    def _segmentation(self, feature_statistics_df):
        """
        perform segmentation of a given audio signal's feature statistics dataframe into “speech” and “music”.

        parameters:
            feature_statistics_df : pandas.DataFrame, shape=(n_segments, m_features)
        
        return:
            Db : np.ndarray, shape=(n_segments,), -1 or 1, -1 -> music, 1 -> speech
        """
        # number of features above its corresponding extrem speech threshold
        S_ex_left = np.where((feature_statistics_df[self.es_features].values - self.es_thresholds_left) < 0, 1, 0)
        S_ex_right = np.where((feature_statistics_df[self.es_features].values - self.es_thresholds_right) > 0, 1, 0)
        S_x = np.sum(S_ex_left + S_ex_right, axis=-1)

        # number of features above its corresponding high probability speech threshold
        S_h = np.sum(np.where(((feature_statistics_df[self.hs_features].values - self.hs_thresholds) * self.hs_positions) > 0, 1, 0) , axis=-1)

        # number of features in the separation set that are classified as speech
        S_p = np.sum(np.where(((feature_statistics_df[self.sp_features].values - self.sp_thresholds) * self.sp_speech_positions) > 0, 1, 0) , axis=-1)

        # number of features above its corresponding extrem music threshold
        M_ex_left = np.where((feature_statistics_df[self.em_features].values - self.em_thresholds_left) < 0, 1, 0)
        M_ex_right = np.where((feature_statistics_df[self.em_features].values - self.em_thresholds_right) > 0, 1, 0)
        M_x = np.sum(M_ex_left + M_ex_right, axis=-1)

        # number of features above its corresponding high probability music threshold
        M_h = np.sum(np.where(((feature_statistics_df[self.hm_features].values - self.hm_thresholds) * self.hm_positions) > 0, 1, 0) , axis=-1)

         # number of features in the separation set that are classified as music
        M_p = np.sum(np.where(((feature_statistics_df[self.sp_features].values - self.sp_thresholds) * self.sp_music_positions) > 0, 1, 0) , axis=-1)

        # number of segments
        n_segments = feature_statistics_df.count()[0]

        # initial classification label
        Di = np.zeros(n_segments)

        # initial classification for each segment
        for i in range(n_segments):
            Di[i] = self._initial_classification(S_x[i], S_h[i], S_p[i], M_x[i], M_h[i], M_p[i])

        # classification smoothing 
        Ds = np.zeros(n_segments)

        forgetting_factors = np.array([np.e**(-1.0*(self.config['average_period']-k-1)/self.config['time_constant']) for k in range(self.config['average_period'])])
        
        normalizing_constant = forgetting_factors.sum()
        
        for i in range(self.config['average_period']):
            Ds[i] = (forgetting_factors[-1-i:] * Di[:i+1]).sum() / forgetting_factors[-1-i:].sum()
        
        for i in range(self.config['average_period'], n_segments):
            Ds[i] = (forgetting_factors * Di[i-4:i+1]).sum() / normalizing_constant

        # adaptative threshold and binarization
        Db = np.zeros(n_segments)

        T = self.T_init

        Db[0] = 1 if Ds[0] > 0 else -1
        
        for i in range(1, n_segments):
            if Ds[i] >= T:
                Db[i] = 1
            elif Ds[i] =< (-T):
                Db[i] = -1
            else:
                if Ds[i] < Ds[i-1]:
                    Db[i] = -1
                else:
                    Db[i] = 1
            if Db[i] == Db[i-1]:
                T = max(0.9*T, self.T_min)
            else:
                T = self.T_init

        return Db


    def tag(self):
        """
        this function can be called to tag files
        """
        




    def _initial_classification(self, Sx, Sh, Sp, Mx, Mh, Mp):
        """
        excute initial classification given Sx, Sh, Sp, Mx, Mh, Mp, alpha.

        note: the logic of this function is based on Figure 4 in the Decision paper

        parameters:
            Sx : int, number of features above its corresponding extrem speech threshold
            Sh : int, number of features above its corresponding high probability speech threshold
            Sp : int, number of features in the separation set that are classified as speech
            Mx : int, number of features above its corresponding extrem music threshold
            Mh : int, number of features above its corresponding high probability music threshold
            Mp : int, number of features in the separation set that are classified as music
        return 
            resault : float number during (-1, 1), -1: music, 1: speech
        """

        # initialize the alpha, mentioned at page 9 of the paper
        alpha = 0.66666

        if (Sx > 0 and Mx == 0 and Mh == 0) or (Sx > 1 and Mx == 0) or (Sh > alpha*self.setting['n_hs_features'] and Mh == 0):
            return 1.0
        elif (Mx > 0 and Sx == 0 and Sx == 0) or (Mx > 1 and Sx == 0) or (Mh > alpha*self.setting['n_hm_features'] and Sh == 0):
            return -1.0
        else:
            resault = 1.0 * (Sp - Mp) / self.setting['n_sp_features']
            return resault

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hc:s:',['help', 'config=', 'sys='])
    except getopt.GetoptError as e:
        print('python Target.py -s <system setting file path> -c <config file path>')

    for opt, value in opts:
        if opt in ['-h', '--help']:
            print('python Target.py -s <system setting file path> -c <config file path>')
        elif opt in ['-s', '--sys']:
            system = value
        elif opt in ['-c', '--config']:
            cfg = value
    try:
        Test(system, cfg)
    except UnboundLocalError as e:
        print('python Target.py -s <system setting file path> -c <config file path>')
    except Exception as e:
        raise(e)

if __name__ == '__main__':
    main(sys.argv[1:])