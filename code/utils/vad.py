import numpy
from scipy.io import wavfile
import os

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
    sig = sig[:, numpy.newaxis]
  # Manage padding
  c = (context, ) + (sig.ndim - 1) * ((0, 0), )
  _win_size = win_size + sum(context)
  shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
  strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
  if pad == 'zeros':
    return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                              shape=shape,
                                              strides=strides).squeeze()
  elif pad == 'edge':
    return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'edge'),
                                              shape=shape,
                                              strides=strides).squeeze()

def pre_emphasis(input_sig, pre):
  """
  Pre-emphasis of an audio signal.
  :param input_sig: the input vector of signal to pre emphasize
  :param pre: value that defines the pre-emphasis filter. 
  """
  if input_sig.ndim == 1:
    return (input_sig - numpy.c_[input_sig[numpy.newaxis, :][..., :1],
                                     input_sig[numpy.newaxis, :][..., :-1]].squeeze() * pre)
  else:
    return input_sig - numpy.c_[input_sig[..., :1], input_sig[..., :-1]] * pre


def vad_percentil(log_energy, percent):
    """

    :param log_energy:
    :param percent:
    :return:
    """
    thr = numpy.percentile(log_energy, percent)
    return log_energy > thr, thr

def vad(filepath, labelpath, lab):
  """
  vad
  params:
    filepath : 
    labelpath :
    label : the label you want to tag
  """
  fs, data = wavfile.read(filepath)
  if len(data.shape) == 2:
    data = data.mean(axis=-1)

  framed = framing(data, win_size=int(fs*0.025), win_shift=int(fs*0.01)).copy()
  # Pre-emphasis filtering is applied after framing to be consistent with stream processing
  framed = pre_emphasis(framed, 0.97)

  log_energy = numpy.log((framed**2).sum(axis=1)+0.000000001)

  label, threshold = vad_percentil(log_energy, 27)

  if len(label) < len(log_energy):
    label = numpy.hstack((label, numpy.zeros(len(log_energy)-len(label), dtype='bool')))

  label_copy = numpy.copy(label)
  filter_context = 10
  for i in range(label.size):
    label[i] = numpy.median(label_copy[max(0, i-filter_context ) : min(i+filter_context, label.size-1)])
  
  label_content = ""
  label = numpy.append(label, False)
  label = numpy.insert(label, 0, False)
  start = 0
  for i in range(1, label.size):
    if label[i] and not label[i-1]:
      start = i
    if not label[i] and label[i-1]:
      label_content += "{} {} {}\n".format((start-1)*0.01, (i-1)*0.01, lab)
  with open(labelpath, 'w') as f:
    f.write(label_content)

  return

