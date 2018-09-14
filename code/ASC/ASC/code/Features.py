# -*- coding: utf-8 -*-
#/python3.5
import librosa as lrs
import os
import numpy as np
import scipy.fftpack as fft
import builtins
import math

from Config import Config

class Feature(builtins.object):
    """
    return a instance that can caculate kinds of audio features,
    this class just can handle with 16000hz and mono wav files.
    
    self._config
    self._data
    self._frame_length
    self._frame_hop_length
    self._segment_length
    self._segment_hop_length
    self._nframes_asegment
    self._n_fft
    self._mfcc_order
    self._roll_percent


    """
    def __init__(self, src, cfg):
        """
        Parameters:
            src : string; wave file path
            cfg : string; config file path. set frame length, frame shift, segment length and so on
        """

        # load the config file
        self._config = Config(cfg).cfgdic

        self.init()

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
        self._data, _ = lrs.load(src, sr=self._operating_rate, mono=True, dtype=np.float64)



    def init(self):
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

        total_energy = np.sum(stft_square[:,1:]) + 0.000001 # avoid being diveded by zero
        low_energy = np.sum(stft_square[:, 1 : int(low_band_bound/self._operating_rate*self._n_fft)], axis=-1)
        high_energy = np.sum(stft_square[:, int(high_band_bound/self._operating_rate*self._n_fft):], axis=-1)

        low = 10 * np.log10(low_energy / total_energy + 1)
        high = 10 * np.log10(high_energy / total_energy + 1)

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
        norm_factors = np.sum(framed_data**2, axis=-1) + 0.000001

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
        mfcc_diff_norm = mfcc_diff / (mfcc_diff_norm_factor + 0.000001)

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

        flux_matrix = stft_matrix[1:,:] - stft_matrix[0:stft_matrix.shape[0]-1,:]

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

        # compute the sum of fft square for each frame, plus 0.000001 to avoid computing error
        stft_square_sum = np.sum(stft_square, axis=-1) + 0.000001

        # compute the center frequencies of each frequence bin and discard 0 frequency
        frequencies_bin = lrs.core.fft_frequencies(sr=self._operating_rate, n_fft=self._n_fft)[1:]

        # compute the perceptually adapted audio spectral centroid for each frame
        ASC = np.sum(np.log2(frequencies_bin/1000 * stft_square + 0.000001), axis=-1) / stft_square_sum

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
            skewness = np.mean((feature_segmented-mean.reshape(-1,1))**3, axis=-1) / std**3
            skewness_diff = np.mean((feature_diff_segmented-mean_diff.reshape(-1,1))**3, axis=-1) / std_diff**3

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

    def _test(self):
        print('*******************************************************************************************************')
        print('short time energy test, shape: ', self._short_time_energy().shape)
        print('*******************************************************************************************************')
        print('zero crossing rate test, shape: ', self._zero_crossing_rate().shape)
        print('*******************************************************************************************************')
        print('band energy ratio test, shape: ', self._band_energy_ratio()[0].shape, self._band_energy_ratio()[1].shape)
        print('*******************************************************************************************************')
        print('autocorrelation coefficient test, shape: ', self._autocorrelation_coefficient().shape)
        print('*******************************************************************************************************')
        print('mfcc test, shape: ', self._mfcc()[0].shape, self._mfcc()[1].shape)
        print('*******************************************************************************************************')
        print('spectrum roll off test, shape; ', self._spectrum_rolloff().shape)
        print('*******************************************************************************************************')
        print('spectral centroid test, shape: ', self._spectrum_centroid().shape)
        print('*******************************************************************************************************')
        print('spectral flux test, shape: ', self._spectral_flux().shape)
        print('*******************************************************************************************************')
        print('spectral spread test, shape: ', self._spectrum_spread().shape)
        print('*******************************************************************************************************')

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
