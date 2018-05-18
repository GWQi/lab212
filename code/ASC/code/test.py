import librosa as lrs
from Learn import Train

obj = Train('/home/guwenqi/Documents/test.wav', '/guwenqi/guwenqi/code/ASC/cfg/Train.cfg')

# print(obj._nframes_asegment, obj._nframes_asegment_hop)

mean, std, mean_diff, std_diff = obj._short_time_energy_statistics()
print('--------------------------------------------------------')
print('short time energy statistics: ', mean.shape, std.shape, mean_diff.shape, std_diff.shape)


mean, std, skewness, mean_diff, std_diff, skewness_diff = obj._zero_crossing_rate_statistics()
print('--------------------------------------------------------')
print('zero crossing rate statistics: ', mean.shape, std.shape, skewness.shape, mean_diff.shape, std_diff.shape, skewness_diff.shape)


low_mean, low_std, low_mean_diff, low_std_diff, high_mean, high_std, high_mean_diff, high_std_diff = obj._band_energy_ratio_statistics()
print('--------------------------------------------------------')
print('band energy ratio statistics: ', low_mean.shape, low_std.shape, low_mean_diff.shape, low_std_diff.shape, high_mean.shape, high_std.shape, high_mean_diff.shape, high_std_diff.shape)


mean, std, mean_diff, std_diff = obj._autocorrelation_coefficient_statistics()
print('--------------------------------------------------------')
print('autocorrelation coefficients statistics: ', mean.shape, std.shape, mean_diff.shape, std_diff.shape)


mfcc_mean, mfcc_std, mfcc_mean_diff, mfcc_std_diff, mfcc_diff_norm_mean, mfcc_diff_norm_std, mfcc_diff_norm_mean_diff, mfcc_diff_norm_std_diff = obj._mfcc_statistics()
print('--------------------------------------------------------')
print('mfcc statistics: ', mfcc_mean.shape, mfcc_std.shape, mfcc_mean_diff.shape, mfcc_std_diff.shape, mfcc_diff_norm_mean.shape, mfcc_diff_norm_std.shape, mfcc_diff_norm_mean_diff.shape, mfcc_diff_norm_std_diff.shape)


mean, std, mean_diff, std_diff = obj._spectrum_rolloff_statistics()
print('--------------------------------------------------------')
print('spectrum rolloff statistics: ', mean.shape, std.shape, mean_diff.shape, std_diff.shape)


mean, std, mean_diff, std_diff = obj._spectrum_centroid_statistics()
print('--------------------------------------------------------')
print('spectrum centroid statistics: ', mean.shape, std.shape, mean_diff.shape, std_diff.shape)


mean, std, mean_diff, std_diff = obj._spectral_flux_statistics()
print('--------------------------------------------------------')
print('spectrum flux statistics: ', mean.shape, std.shape, mean_diff.shape, std_diff.shape)


mean, std, mean_diff, std_diff = obj._spectrum_spread_statistics()
print('--------------------------------------------------------')
print('spectrum spread statistics: ', mean.shape, std.shape, mean_diff.shape, std_diff.shape)

LSTER = obj._LSTER()
print('--------------------------------------------------------')
print('LSTER: ', LSTER.shape)

feature_statistics = obj._feature_statistics_concate()
print('--------------------------------------------------------')
print('features statistics concation: ', feature_statistics.shape)