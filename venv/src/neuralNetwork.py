import numpy as np
import glob, os
from layers import _FullyConnected, _Input, _deriv_softmax, FUNCTIONS

"""    ---------------------    Neural Network      ---------------------   """
""" For vector and matrix variables which are arguments for below API, use numpy structures """
class Neural:
  def __init__(self):
    self.layers = []

  def save_network(self, directory):
    """ Saves network in passed directory. Later you can load it by 'load_network' method. """
    for f in glob.glob("./network/weights*.csv"):
        os.remove(f)
    with open(directory + '/structure', "w") as f:
      f.write("Input " + str(self.layers[0].get_neurons_number()) + '\n')
      for i in range(1, len(self.layers)):
        f.write("Fully " + str(self.layers[i].get_neurons_number()) + ' ' +
                self.layers[i].activation_name + '\n')
        np.savetxt(directory + '/weights' + str(i) + '.csv', self.layers[i].get_weights(), delimiter=',')

  def load_network(self, directory):
    """ Load network structure and weights from directory/files to model """
    with open(directory + '/structure', "r") as f:
      line = f.readline()
      input_name, input_size = line.split(' ')
      self.add_input(int(input_size))
      i = 0

      while 1:
        line = f.readline()
        if len(line) <= 0:
          break
        i += 1
        layer_name, layer_size, layer_activation = line.split(' ')
        layer_activation = layer_activation[:-1]  # get rid of endline
        self.add_fully_connected(int(layer_size), layer_activation)
        self.layers[-1].set_weights(np.loadtxt(directory + '/weights' + str(i) + '.csv', delimiter=','))

  def add_input(self, size):
    """ Adds input layer to model, it must be used before any add_* function.
        Size indicate to size of input data. Return True if adding layer has been
        completed with success, otherwise False."""
    if size <= 0:
      print("Error: Input size must be >0")
      return False
    self.layers = [_Input(size)]
    return True

  def add_fully_connected(self, neurons, activation='linear', left_range=-0.1, right_range=0.1):
    """ Adds fully connected layer. Return True if adding layer has been
        completed with success, otherwise False. """
    if neurons <= 0 or activation not in FUNCTIONS:
      print("Error: Neurons number must be >0 and activation must be in FUNCTIONS from layers.py")
      return False
    self.layers.append(_FullyConnected(self.layers[-1], neurons, activation, left_range, right_range))
    return True

  def predict(self, input_data):
    """ If passed 'input_data' is valid to network structure it returns
        network answer, otherwise None. """
    if not self._is_valid(input_data):
      self._print_error_message(input_data)
      return None

    output = input_data
    for i in range(1, len(self.layers)):
      output = output @ np.transpose(self.layers[i].get_weights())
      output = self.layers[i].activation(output)
    return output

  def fit(self, input_data, expected_data, iterations=100, alpha=0.01, drop_percent=0.5, test_input=None, test_expected=None):
    """ 'input_data' are passed as tuple (number of input, size of single input), 'expected_data' is
        tuple (number of expected, size of single expected) as well. 'iterations' (>1) indicate how many times run
        training. 'alpha' (0.0-1.0) is learning rate, which says how fast learn network. Good
        adjustment of this parameter results in better training. 'drop_percent' (0.0-1.0) prevents
        overfitting network to training data and makes it more general. Literally it
        says how much neurons in hidden layer should we 'turn off'.
        Return True if training has been completed with success, otherwise False."""
    if not self._is_valid(input_data, expected_data):
      self._print_error_message(input_data, expected_data)
      return False
    batch_size = input_data.shape[0]
    self._print_training_info(input_data, iterations, alpha, drop_percent)

    for it in range(iterations):
      dropout_mask, scaled_dropout_mask = self._create_dropout_mask(batch_size, drop_percent)
      layers_output = self._get_layers_output(input_data, dropout_mask, scaled_dropout_mask)
      layers_delta = self._get_delta(batch_size, layers_output, expected_data, dropout_mask)
      layers_weighted_delta = self._get_weighted_delta(layers_output, layers_delta)
      self._update_weights(layers_weighted_delta, alpha)
      self._test(test_input, test_expected, it, iterations)   # just for testing purposes (can be deleted)

    return True

  def _is_valid(self, input_data, expected_data=None):
    """ Return True if data is valid, otherwise False """
    if expected_data is None:
      return len(input_data.shape) == 2 and input_data.shape[1] == self.layers[0].get_neurons_number()
    else:
      return len(input_data.shape) == 2 and len(expected_data.shape) == 2 and \
             input_data.shape[0] == expected_data.shape[0] and \
             input_data.shape[1] == self.layers[0].get_neurons_number() and \
             expected_data.shape[1] == self.layers[-1].get_neurons_number()

  def _create_dropout_mask(self, batch_size, drop_percent):
    """If there is no hidden layers (number of layers < 3) dropout mask is not used and
       (None, False) is returned, where False says that dropout is not used.
       Otherwise it creates dropout mask of shape ('batch_size', first_hidden_layer_size)
       and scaled dropout mask depend on 'dropout_percent' parameter.
       Then it return (scaled dropout mask, True), where True says that dropout used in training."""
    if len(self.layers) < 3:
      return None, None

    dropout_mask = np.zeros((batch_size, self.layers[1].get_neurons_number()))
    for vector in dropout_mask:
      vector[:int(len(vector) * drop_percent)] = 1.0
      np.random.shuffle(vector)
    ratio = 1.0 / drop_percent
    scaled_dropout_mask = dropout_mask * ratio
    return dropout_mask, scaled_dropout_mask

  def _get_layers_output(self, input_data, dropout_mask, scaled_dropout_mask):
    """ Return output of each layer in network based on given 'input_data' modified,
        if possible, by 'scaled_dropout_mask' """
    layers_output = [input_data]
    for lay_it in range(1, len(self.layers)):
      layers_output.append(layers_output[lay_it - 1] @ np.transpose(self.layers[lay_it].get_weights()))
      layers_output[lay_it] = self.layers[lay_it].activation(layers_output[lay_it])
      if lay_it == 1 and dropout_mask is not None:
        layers_output[lay_it] *= scaled_dropout_mask

    return layers_output

  def _get_delta(self, batch_size, layers_output, expected_data, dropout_mask):
    """ Prepare and return delta values for each layer basing on given parameters. """
    layers_delta = [0.0 for i in range(len(self.layers))]   # delta for input layer is initialized, but not used
    if self.layers[-1].derivative == _deriv_softmax:
      layers_delta[-1] = self.layers[-1].derivative(layers_output[-1], expected_data)
    else:
      layers_delta[-1] = layers_output[-1] - expected_data
    layers_delta[-1] /= batch_size

    for lay_it in range(len(self.layers) - 2, 0, -1):
      layers_delta[lay_it] = layers_delta[lay_it + 1] @ self.layers[lay_it+1].get_weights()
      layers_delta[lay_it] *= self.layers[lay_it].derivative(layers_output[lay_it])
      if (lay_it == 1 and dropout_mask is not None):
        layers_delta[lay_it] *= dropout_mask

    return layers_delta

  def _get_weighted_delta(self, layers_output, layers_delta):
    """ Return delta weighted by layers output """
    layers_weighted_delta = [0 for i in range(len(self.layers))]  # weighted delta for input layer is initialized, but not used
    for lay_it in range(len(self.layers) - 1, 0, -1):
      layers_weighted_delta[lay_it] = np.transpose(layers_delta[lay_it]) @ layers_output[lay_it - 1]

    return layers_weighted_delta

  def _update_weights(self, layers_weighted_delta, alpha):
    """ Update layers weights basing on weighted delta multiplied by alpha """
    for lay_it in range(len(self.layers) - 1, 0, -1):
      weights = self.layers[lay_it].get_weights()
      weights -= layers_weighted_delta[lay_it] * alpha

  def _print_error_message(self, input_data, expected_data=None):
    print("Error: Input data or expected data are incorrect.")
    print("   Shapes of passed data and network structure does not match.")
    print("       Passed input:", input_data.shape)
    if expected_data is None:
      print("       Passed expected: None")
    else:
      print("       Passed expected:", expected_data.shape)
    print("       Network input:", self.layers[0].get_neurons_number())
    print("       Network output:", self.layers[-1].get_neurons_number(), "\n")

  def _print_training_info(self, input_data, iterations, alpha, drop_percent):
    print("Info:   Training parameters")
    print("   Number of training data:", input_data.shape[0])
    print("   Iterations:", iterations)
    print("   Alpha:", alpha)
    print("   Dropout percent:", drop_percent, "\n")

  def _test(self, input_data, expected_data, it=0, iterations=0):
    """ Temporary method for testing purposes """
    if (it % (int(iterations / 10))) != 0 or input_data is None or expected_data is None:
      return
    if not self._is_valid(input_data, expected_data):
      self._print_error_message(input_data, expected_data)
      return
    prediction = self.predict(input_data)
    correct = 0
    error = 0
    for i in range(len(prediction)):
        net_answer = np.argmax(prediction[i])
        expected_answer = np.argmax(expected_data[i])
        correct += (net_answer == expected_answer)
        error += (net_answer - expected_answer)**2
    error /= input_data.shape[0]*input_data.shape[1]
    print("Iteration [%d]. accuracy: %.2f  error %.6f" % (it, ((correct/len(prediction))*100), error))
