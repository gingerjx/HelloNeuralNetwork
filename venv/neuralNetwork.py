import random
import numpy as np
import activationFunctions as af

class DeepNetwork:
  def __init__(self, layers, leftRange=-0.1, rightRange=0.1, weights=None):
    self.layers = layers                                    # structure of net
    self.actFuncs = [af.ReLU] * (len(layers)-1)             # by default ReLU activation function for each hidden layer
    self.derivFuncs = [af.derivReLU] * (len(layers)-1)      # and its derivative
    self.actFuncs[-1] = af.emptyFunc                        # by default, there's no change in output layer
    self.derivFuncs[-1] = af.emptyFunc                      # like above
    self.enableSoftmax = False                              # enable softmax functions as AF for output layer
    self.weights = []                                       # weights[layer][neuron][weight]

    if weights is not None:
      self.weights = weights
    else:
      for i in range(len(layers)-1):
        weights2D = []
        for j in range(layers[i+1]):
          weights1D = []
          for k in range(layers[i]):
            weightValue = np.random.uniform(leftRange,rightRange)
            weights1D.append(weightValue)
          weights2D.append(weights1D)
        self.weights.append(np.array(weights2D))

  def setHiddenAF(self,indexLayer,actFunction,derivFunction):
      # -- Only hidden layers are able to be set in here -- #
      assert indexLayer < len(self.layers)-2
      self.actFuncs[indexLayer] = actFunction
      self.derivFuncs[indexLayer] = derivFunction

  def setSoftmax(self,bool):
      # -- Only option is to set outputlayer AF as softmax or empty -- #
      self.enableSoftmax = bool
      if bool == True:
          self.actFuncs[-1] = af.softmax
      else: self.actFuncs[-1] = af.emptyFunc

  def predict(self,inputData):
    output = inputData
    for i in range(len(self.weights)):
      output = np.matmul(output, np.transpose(self.weights[i]))
      output = self.actFuncs[i](xxx=output)
    return output

  def fit(self, inputData, expectedData, iterations=100, alpha=0.01, batchSize=-1, dropPercent=0.5, testInput=None, testExpected=None):
    assert inputData.shape[0] == expectedData.shape[0]
    assert batchSize <= inputData.shape[0]
    if batchSize == -1:
      batchSize = inputData.shape[0]
    print("Training parameters - iterations: %d   alpha: %lf    batchSize: %ld    dropPercent: %f" %
                                (iterations, alpha, batchSize, dropPercent) )
    accuracy = 0

# -- Start iterating -- #
    for it in range(iterations):
      for bIt in range( int(inputData.shape[0]/batchSize) ):

# -- create batch -- #
        batchInput = []
        batchExpected = []
        if (bIt+1)*batchSize > inputData.shape[0]:
          batchInput = inputData[bIt * batchSize:]
          batchExpected = expectedData[bIt * batchSize:]
        else:
          batchInput = inputData[bIt*batchSize:(bIt+1)*batchSize]
          batchExpected = expectedData[bIt * batchSize:(bIt + 1) * batchSize]

# -- create dropout mask -- #
        droputMask = np.zeros((batchInput.shape[0], self.layers[1]))  # create dropout mask for each input serie for first hidden layer
        for vector in droputMask:
          for i in range(int(len(vector) * dropPercent)):
            vector[i] = 1.0
          np.random.shuffle(vector)
        ratio = 1.0 / dropPercent
        scaledDroputMask = droputMask * ratio                         # crete dropout mask scaled by ratio

# -- calculating layers outputs -- #
        layersOutputs = [batchInput]

        for layIt in range(1, len(self.layers)):
          layersOutputs.append(layersOutputs[layIt - 1] @ np.transpose(self.weights[layIt - 1])) # lay_n_values = lay_(n-1)_values * trans_n_weights
          layersOutputs[layIt] = self.actFuncs[layIt - 1](xxx=layersOutputs[layIt])              # activationFunc(lay_n_values)
          if (layIt == 1 and len(self.layers) > 2):
            layersOutputs[layIt] *= scaledDroputMask                                             # scaledDroputMask(lay_n_values)

# -- calculating delta -- #
        layersDelta = [0 for i in range(len(self.layers))]                                  # input layer delta declared, but not used
        layersDelta[-1] = layersOutputs[-1] - batchExpected                                 # layer_out_delta = actFunc(layer_out_values - expected_output)
        if self.enableSoftmax:
            layersDelta[-1] = np.array(af.derivSoftmax(xxx=layersOutputs[-1],eee=batchExpected))    # layer_out_delta = layer_out_delta * derivActFunc(layer_out_delta)
        layersDelta[-1] /= batchSize                                                        # dividing output delta by batch size

        for layIt in range(len(self.layers) - 2, 0, -1):
          layersDelta[layIt] = layersDelta[layIt + 1] @ self.weights[layIt]                 # layer_n_delta = layer_(n+1)_delta * layer_(n+1)_weights
          layersDelta[layIt] *= self.derivFuncs[layIt - 1](xxx=layersOutputs[layIt])        # layer_n_delta = layer_n_delta * derivActFunc(layer_n_delta)
          if (layIt == 1 and len(self.layers) > 2):
            layersDelta[layIt] *= droputMask                                                # dropoutMask(layer_n_delta)

# -- calculating weighted -- #
        layersWeightedDelta = [0 for i in range(len(self.layers))]                                 # input layer weighted delta declared, but not used
        for layIt in range(len(self.layers) - 1, 0, -1):
          layersWeightedDelta[layIt] = np.transpose(layersDelta[layIt]) @ layersOutputs[layIt - 1] # layer_n_weighted_delta = trans(layer_n_delta)*layer_(n-1)_values

# -- updating weights -- #
        for layIt in range(len(self.layers) - 1, 0, -1):
          self.weights[layIt - 1] -= layersWeightedDelta[layIt] * alpha   # layer_n_weights = layer_n_weights * alpha * layer_n_weighted_delta

# -- testing network -- #
      if (it % (int(iterations / 10))) == 0 and iterations > 10:
        self.test(inputData, expectedData, it, "Training")
        if testInput is not None and testExpected is not None:
          newAccuracy = self.test(testInput, testExpected, it, "Test")
          if newAccuracy < accuracy:
            print("Net overfitted\n")
            break
          else:
            accuracy = newAccuracy
            print("")

  def test(self,inputData,expectedData,iter=0,text="Test"):
    assert inputData.shape[0] == expectedData.shape[0]
    prediction = self.predict(inputData)
    correct = error = 0
    for i in range(len(prediction)):
        assert sum(prediction[i]) > 0.99 and sum(prediction[i]) < 1.01  #temporary
        netAnswer = np.argmax(prediction[i])
        expectedAnswer = np.argmax(expectedData[i])
        correct += (netAnswer == expectedAnswer)
        error += (netAnswer - expectedAnswer)**2
    error /= inputData.shape[0]*inputData.shape[1]
    print("[%d] %s accuracy: %.2f  error %.6f" % (iter, text, ((correct/len(prediction))*100), error))
    return (correct/len(prediction))*100

