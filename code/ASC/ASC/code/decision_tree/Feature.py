import librosa as lrs

def short_time_energy(data, frame_length, frame_hop_length):
    """
    caculate short time energy for each frame
        
    Returns:
        ste : np.ndarray [shape=(n_frames,)]; ste[i] is the fraction of short time energy in the i th frame
    """

    framed_data = lrs.util.frame(data,\
                                 frame_length=frame_length,\
                                 hop_length=frame_hop_length).transpose()
    ste = 10 * np.log10(1.0 / frame_length * np.sum(framed_data**2, axis=-1) + 1)

    return ste

def zero_crossing_rate(data, frame_length, frame_hop_length):
    """
    caculate zero crossing rate for each frame

    Returns:    
        zcr : np.ndarray [shape=(n_frames,)]; zcr[i] is the fraction of zero crossings in the i th frame

    """
    zcr = lrs.feature.zero_crossing_rate(data,\
                                         frame_length=frame_length,\
                                         hop_length=frame_hop_length,\
                                         center=False)[0]
    return zcr

def band_energy_ratio(self):
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