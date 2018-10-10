
import numpy as np
from scipy.io import wavfile

def framing(sig, win_size, win_shift, context=(0, 0), pad='zeros'):
  """
  :param sig: input signal, can be mono or multi dimensional
  :param win_size: size of the window in term of samples
  :param win_shift: shift of the sliding window in terme of samples
  :param context: tuple of left and right context
  :param pad: can be zeros or edge
  """
  dsize = sig.dtype.itemsize
  if sig.ndim == 1:
    sig = sig[:, np.newaxis]
  # Manage padding
  c = (context, ) + (sig.ndim - 1) * ((0, 0), )
  _win_size = win_size + sum(context)
  shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
  strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
  if pad == 'zeros':
    return np.lib.stride_tricks.as_strided(np.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                              shape=shape,
                                              strides=strides).squeeze()
  elif pad == 'edge':
    return np.lib.stride_tricks.as_strided(np.lib.pad(sig, c, 'edge'),
                                              shape=shape,
                                              strides=strides).squeeze()

def pre_emphasis(input_sig, pre):
  """
  Pre-emphasis of an audio signal.
  :param input_sig: the input vector of signal to pre emphasize
  :param pre: value that defines the pre-emphasis filter. 
  """
  if input_sig.ndim == 1:
    return (input_sig - np.c_[input_sig[np.newaxis, :][..., :1],
                                     input_sig[np.newaxis, :][..., :-1]].squeeze() * pre)
  else:
    return input_sig - np.c_[input_sig[..., :1], input_sig[..., :-1]] * pre


def vad_percentil(log_energy, percent):
    """

    :param log_energy:
    :param percent:
    :return:
    """
    thr = np.percentile(log_energy, percent)
    return log_energy > thr, thr

def vad(data, sr=16000, percent=27.0):
  """
  vad
  params:
    data : wav data
    sr : int, sample rate of this data
    percent : int, energy detect threshold
  return:
    signal : np.ndarray, signal decided to be voice
    indexes : indexes where the samples are not silence
  """

  framed = framing(data, win_size=int(sr*0.025), win_shift=int(sr*0.01)).copy()
  # Pre-emphasis filtering is applied after framing to be consistent with stream processing
  framed = pre_emphasis(framed, 0.97)
  # print("framed.size: ", framed.size)

  log_energy = np.log((framed**2).sum(axis=1)+0.000000000000001)
  # print("log_energy.size: ", log_energy.size)

  label, threshold = vad_percentil(log_energy, percent)

  if len(label) < len(log_energy):
    label = np.hstack((label, np.zeros(len(log_energy)-len(label), dtype='bool')))

  label_copy = np.copy(label)
  filter_context = 10
  for i in range(label.size):
    label[i] = np.median(label_copy[max(0, i-filter_context ) : min(i+filter_context, label.size-1)])
  
  label = np.append(label, False)
  label = np.insert(label, 0, False)

  start = 0
  signal = np.zeros(0, dtype=data.dtype)
  for i in range(1, label.size):
    if label[i] and not label[i-1]:
      start = i
    if not label[i] and label[i-1]:
      signal = np.append(signal, data[(start-1)*sr:(i-1)*sr])

  return signal

