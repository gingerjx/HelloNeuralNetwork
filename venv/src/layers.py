import numpy as np

"""    ---------------------    Activation Functions      ---------------------   """
"""  'e' parameter in derivative functions are not used except 'deriv_softmax'.
     Cause of need of this parameter in 'deriv_softmax' and convenience in coding
     it is put in other derivative functions."""
def _linear(x, e=None):
  return x

def _relu(x):
  return x * (x > 0)
def _deriv_relu(x, e=None):
  return np.ones(x.shape) * (x > 0)

def _tanh(x):
  return np.tanh(x)
def _deriv_tanh(x, e=None):
  return 1.0 - np.tanh(x) ** 2

def _softmax(x):
  return np.exp(x) / sum(np.exp(x))
def _deriv_softmax(x, e):
  if len(x.shape) != 3:
    return (x - e) / len(e)
  else:
    return [_deriv_softmax(x[i], e[i]) for i in range(len(x))]

""" If you are adding some activation function, add it here as well with its derivative """
FUNCTIONS = {'linear': (_linear, _linear),
             'relu': (_relu, _deriv_relu),
             'tanh': (_tanh, _deriv_tanh),
             'softmax': (_softmax, _deriv_softmax)}

"""    ---------------------    Layers      ---------------------   """
class _Input:
  """ Simple input layer, contain information about input data size """
  def __init__(self, size):
    self.size = size

  def get_neurons_number(self):
    """ Return number of neurons in layers """
    return self.size

class _FullyConnected:
  """ Fully connected layers is kind of layer, where all neurons in adjacent layers
      are connected with each other."""
  def __init__(self, prev_layer, neurons_number, activation, left_range, right_range):
    self.activation_name = activation   # convenience during saving network
    self.activation = FUNCTIONS[activation][0]
    self.derivative = FUNCTIONS[activation][1]
    self.neurons_number = neurons_number
    self.weights = []

    for i in range(neurons_number):
      self.weights.append([np.random.uniform(left_range, right_range)
                           for j in range(prev_layer.get_neurons_number())])
    self.weights = np.array(self.weights)

  def get_neurons_number(self):
    """ Return number of neurons in layers. """
    return self.neurons_number

  def get_weights(self):
    """ Return original weights. Modifying AFFECT on original. """
    return self.weights

  def get_weights_copy(self):
    """ Return copy of weights. Modifying HAVE NO affect on original. """
    return self.weights.copy()

  def set_weights(self, weights):
    """ Set layer's weights """
    self.weights = weights