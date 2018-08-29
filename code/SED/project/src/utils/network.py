from __feature__ import print_function

import tensorflow as tf
import sys

def conv2d(inputs, kernel_size, outchannel, pool_strides, name):
  """
  create a conv2d layer
  params:
    inputs : tf.Tensor, shape=(batch, hight, width, channel)
    knernel_size : list of int, [vertical_size, horizonal_size]
    outchannel : int, number of outputs channel
    pool_strides : list of lists of int, [vertical_stride, horizonal_stride]
    name : string, name prefix of this conv layer

  Note: if you want to apply batch normalization to CNN, add it in this function
  """
  inputs_shape = tf.shape(inputs)
  inchannel = shape[-1]
  kernel_shape = list(kernel_size) + [inchannel, outchannel]

  w = tf.get_variable(name=name+"/w", shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable(name=name+"/b", shape=[outchannel], initializer=tf.zeros_initializer())
  h_conv = tf.nn.relu(tf.add(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding='SAME'), b))
  h_pool = tf.nn.max_pool(h_conv, [1, 2, 2, 1], [1] + pool_strides + [1], padding'SAME')

  return h_pool

def multiLayerConv2d(num_layers, inputs, kernels_size, outchannels, pool_strides, name):
  """
  this function is used to build multiple convolution layers
  params:
    inputs : tf.Tensor, shape=(batch, hight, width, channel)
    num_layers : int, number of layers
    kernels_size : list of lists, which contain the kernek_size of each layer.
    outchannels : list of int, out channels of each layer
    pool_strides : list of lists of int, pooling max strides of each layer
    name : string, name prefix of this multiply convolutional layers

  Note: if you want to apply batch normalization to CNN, add it in conv2d function
  """
  if len(kernels_size) != num_layers or not isinstance(kernels_size[0], list):
    print("Fatal error at function multiLayerConv2d, argv knernel_size must be:[[v1, h1], [v2, h2], ...], and length of kernels_size must be same of num_layers")
    sys.exit(0)
  if len(outchannels) != num_layers:
    print("Fatal error at function multiLayerConv2d, length of argv outchannels must be same as num_layers.")
    sys.exit(0)
  if len(pool_strides) != num_layers or not isinstance(pool_strides[0], list):
    print("Fatal error at function multiLayerConv2d, argv pool_strides must be:[[v1, h1], [v2, h2], ...], and length of pool_strides must be same of num_layers")

  for i in range(num_layers):
    inputs = conv2d(inputs, kernels_size[i], outchannels[i], pool_strides[i], name + '_' +str(i+1))

  return inputs

def FNNLayer(inputs, units, relu_clip, keep_prob, name):
  """
  this function is used to build a one layer FNN
  params:
    inputs : tf.Tensor, inputs of this layer
    units : int, number of units of this layer
    relu_clip : tf.float32 or float of python build in type
    keep_prob : list of float, keep probility used for dropout
    name : name prefix of this layer
  """
  shape = tf.shape(inputs)
  last_layer_units = shape[-1]
  w = tf.get_variable(name=name+'/w', shape=[last_layer_units, units], initializer=tf.contrib.layer.xavier_initializer())
  b = tf.get_variable(name=name+'/b', shape=[int(units)], initializer=tf.zeros_initializer())
  h = tf.minimum(tf.nn.relu(tf.add(tf.matmul(inputs, w), b)), relu_clip)
  h_drop = tf.nn.dropout(h, keep_prob)

  return h_drop

def multiFNNLayers(num_layers, inputs, units_list, relu_clip, keep_prob_list, name):
  """
  this function is used to construct multi layers FNN
  params:
    num_layers : number of layers
    inputs : tf.Tensor of tf.placeholder
    units_list : list of int, units of each hidden layer
    relu_clip : relu clip value
    keep_prob_list : list of float, used for dropout of every layer
    name : name prefix of this multi layers
  return:
    return the outputs of last layer 
  """
  # daily check
  if len(units_list) != num_layers:
    print("Fatal Error at function multiFNNLayers, length of argv units_list must be same as num_layers")
    sys.exit(0)
  if len(keep_prob_list) != num_layers:
    print("Fatal Error at function multiFNNLayers, length of argv keep_prob_list must be same as num_layers")
    sys.exit(0)

  for i in range(num_layers):
    inputs = FNNLayer(inputs, units_list[i], relu_clip, keep_prob_list[i], name + '_' + str(i+1))

  return inputs

def uniRNN(num_layers, units_list, keep_prob, cell_type, **kwargs):
  """
  construct multi-layers unidirectional RNN networks, you can use gradient clip and batch normalization layer between LSTM and Dense layers
  params:
    num_layers : int, number of rnn layers
    units_list : list/tuple of int, corresponding to each layer.
                 or just a int. if just int, it units number of all layers is same
    keep_prob  : list/tuple of float, keep probality of each layer
                 or just a float, for all layer
    cell_type  : rnn cell class, before you use those cells, PLEASE BROWSE the DETAILS ON tensorflow 
                 BasicRNNCell(property name is deprecated, no peepholes argv) type cell can be used
                 BasicLSTMCell(property name is deprecatedn no peepholes argv) type cell can be used, it is the basic baseline, for advanced models, use LSTMCell
                 LSTMCell(property name is deprecated, has peepholes argv) type cell can be used, 
                 GRUCell(property name is deprecatedn no peepholes argv) type cell can be used
                 LayerNormBasicLSTMCell(This class adds layer normalization(https://arxiv.org/abs/1607.06450) and recurrent dropout(https://arxiv.org/abs/1603.05118) to a basic LSTM unit) can be used, 
                 doubtful, LSTMBlockCell(http://arxiv.org/abs/1409.2329), Unlike LSTMCell,  this is a monolithic op and should be much faster
                 doubtful, GRUBlockCell
                 LSTMBlockFusedCell(http://arxiv.org/abs/1409.2329, apply dropout correctly) can be used, might do not need dropout, This is an extremely efficient LSTM implementation
  return:
    stack_cell : list of cells, used to construct
  """
  # 
  if isinstance(units_list, int):
    units_list = [units_list] * num_layers
  if isinstance(keep_prob, float) or isinstance(keep_prob, int):
    keep_prob = [min(float(keep_prob), 1.0)] * num_layers

  if isinstance(units_list, list):
    if len(units_list) > num_layers:
      units_list = units_list[0:num_layers]
      print("warning: uniRNN function len(units_list) is bigger than num_layers, we just fetch the first num_layer elements of units_list")
    elif len(units_list) < num_layers:
      units_list += [units_list[-1]] * (num_layers - units_list)
      print("warning: uniRNN function len(units_list) is smaller than num_layers, we extend units_list to length num_layers use units_list[-1]")
    for i in range(num_layers):
      units_list[i] = int(units_list[i])

  if isinstance(keep_prob, list):
    if len(keep_prob) > num_layers:
      keep_prob = keep_prob[0:num_layers]
      print("warning: uniRNN function len(keep_prob) is bigger than num_layers, we just fetch the first num_layer elements of keep_prob")
    elif len(keep_prob) < num_layers:
      keep_prob += [keep_prob[-1]] * (num_layers - len(keep_prob))
      print("warning: uniRNN function len(keep_prob) is smaller than num_layers, we extend keep_prob to length num_layers use keep_prob[-1]")
    for i in range(num_layers):
      keep_prob[i] = min(float(keep_prob[i]), 1.0)

  cells = []

  if cell_type == tf.contrib.rnn.BasicRNNCell or cell_type == tf.contrib.rnn.BasicLSTM or cell_type == tf.contrib.rnn.GRUCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], reuse=kwargs.get("reuse", None))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      cells.append(cell_drop)

  # LSTMCell has peepholes and projection function, but BasicLSTMCell doesn't have
  elif cell_type == tf.contrib.rnn.LSTMCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], use_peepholes=kwargs.get("use_peepholes", True), reuse=kwargs.get("reuse", None))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      cells.append(cell_drop)

  # LayerNormBasicLSTMCell has recurrent dropout and layer normalization, but it doesn't have peepholes
  elif cell_type == tf.contrib.rnn.LayerNormBasicLSTMCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], dropout_keep_prob=keep_prob[i], reuse=kwargs.get("reuse", None))
      cells.append(cell)

  # LSTMBlockFusedCell is an extremely efficient LSTM implementation, can use peepholes
  elif cell_type == tf.contrib.rnn.LSTMBlockFusedCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], use_peephole=kwargs.get("use_peephole", True))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      cells.append(cell)
  else:
    print("This function doesn't support cell type: {}".format(cell_type))
    sys.exit(0)

  stack_cell = tf.contrib.rnn.MultiRNNCell(cells)

  return stack_cell

def BiRNN(num_layers, units_list, keep_prob, cell_type, **kwargs):
  """
  construct multi-layers unidirectional RNN networks, you can use gradient clip and batch normalization layer between LSTM and Dense layers
  params:
    num_layers : int, number of rnn layers
    units_list : list/tuple of int, corresponding to each layer.
                 or just a int. if just int, it units number of all layers is same
    keep_prob  : list/tuple of float, keep probality of each layer
                 or just a float, for all layer
    cell_type  : rnn cell class, before you use those cells, PLEASE BROWSE the DETAILS ON tensorflow 
                 BasicRNNCell(property name is deprecated, no peepholes argv) type cell can be used
                 BasicLSTMCell(property name is deprecatedn no peepholes argv) type cell can be used, it is the basic baseline, for advanced models, use LSTMCell
                 LSTMCell(property name is deprecated, has peepholes argv) type cell can be used, 
                 GRUCell(property name is deprecatedn no peepholes argv) type cell can be used
                 LayerNormBasicLSTMCell(This class adds layer normalization(https://arxiv.org/abs/1607.06450) and recurrent dropout(https://arxiv.org/abs/1603.05118) to a basic LSTM unit) can be used, 
                 doubtful, LSTMBlockCell(http://arxiv.org/abs/1409.2329), Unlike LSTMCell,  this is a monolithic op and should be much faster
                 doubtful, GRUBlockCell
                 LSTMBlockFusedCell(http://arxiv.org/abs/1409.2329, apply dropout correctly) can be used, might do not need dropout, This is an extremely efficient LSTM implementation
  return:
    fw_cells : list of forward direction cells
    bw_cells : list of backward direction cells

  Note: the return of this function can be used by: tf.contrib.rnn.stack_bidirectional_dynamic_rnnn()
  """
  if isinstance(units_list, int):
    units_list = [units_list] * num_layers
  if isinstance(keep_prob, float) or isinstance(keep_prob, int):
    keep_prob = [min(float(keep_prob), 1.0)] * num_layers

  if isinstance(units_list, list):
    if len(units_list) > num_layers:
      units_list = units_list[0:num_layers]
      print("warning: uniRNN function len(units_list) is bigger than num_layers, we just fetch the first num_layer elements of units_list")
    elif len(units_list) < num_layers:
      units_list += [units_list[-1]] * (num_layers - units_list)
      print("warning: uniRNN function len(units_list) is smaller than num_layers, we extend units_list to length num_layers use units_list[-1]")
    for i in range(num_layers):
      units_list[i] = int(units_list[i])

  if isinstance(keep_prob, list):
    if len(keep_prob) > num_layers:
      keep_prob = keep_prob[0:num_layers]
      print("warning: uniRNN function len(keep_prob) is bigger than num_layers, we just fetch the first num_layer elements of keep_prob")
    elif len(keep_prob) < num_layers:
      keep_prob += [keep_prob[-1]] * (num_layers - len(keep_prob))
      print("warning: uniRNN function len(keep_prob) is smaller than num_layers, we extend keep_prob to length num_layers use keep_prob[-1]")
    for i in range(num_layers):
      keep_prob[i] = min(float(keep_prob[i]), 1.0)

  fw_cells = []
  bw_cells = []

  if cell_type == tf.contrib.rnn.BasicRNNCell or cell_type == tf.contrib.rnn.BasicLSTM or cell_type == tf.contrib.rnn.GRUCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], reuse=kwargs.get("reuse", None))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      fw_cells.append(cell_drop)
    for i in range(num_layers):
      cell = cell_type(units_list[i], reuse=kwargs.get("reuse", None))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      bw_cells.append(cell_drop)


  # LSTMCell has peepholes and projection function, but BasicLSTMCell doesn't have
  elif cell_type == tf.contrib.rnn.LSTMCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], use_peepholes=kwargs.get("use_peepholes", True), reuse=kwargs.get("reuse", None))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      fw_cells.append(cell_drop)

    for i in range(num_layers):
      cell = cell_type(units_list[i], use_peepholes=kwargs.get("use_peepholes", True), reuse=kwargs.get("reuse", None))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      bw_cells.append(cell_drop)

  # LayerNormBasicLSTMCell has recurrent dropout and layer normalization, but it doesn't have peepholes
  elif cell_type == tf.contrib.rnn.LayerNormBasicLSTMCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], dropout_keep_prob=keep_prob[i], reuse=kwargs.get("reuse", None))
      fw_cells.append(cell)

    for i in range(num_layers):
      cell = cell_type(units_list[i], dropout_keep_prob=keep_prob[i], reuse=kwargs.get("reuse", None))
      bw_cells.append(cell)

  # LSTMBlockFusedCell is an extremely efficient LSTM implementation, can use peepholes
  elif cell_type == tf.contrib.rnn.LSTMBlockFusedCell:
    for i in range(num_layers):
      cell = cell_type(units_list[i], use_peephole=kwargs.get("use_peephole", True))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      fw_cells.append(cell)

    for i in range(num_layers):
      cell = cell_type(units_list[i], use_peephole=kwargs.get("use_peephole", True))
      cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob[i], output_keep_prob=keep_prob[i])
      bw_cells.append(cell)

  else:
    print("This function doesn't support cell type: {}".format(cell_type))
    sys.exit(0)

  return fw_cells, bw_cells

