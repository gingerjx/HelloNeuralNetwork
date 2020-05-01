# Hello Neural Network
Implementation of Deep Neural Network able to recognize digits from MNIST database
## Libraries
numpy 1.18.2
## API
- <b>\__init__(self,layers,leftRange,rightRange,weights=None)</b>
  - layers - is a list of integers saying about our network structure (e.g. [784,40,20,10] networks structure 784-40-20-10),
  - leftRange, rightRange - range of weight draws.
  - weights - if passed, left and right range are not used and network use this weights


- <b>setHiddenAF(self,indexLayer,actFunction,derivFunction)</b> set activation function of hidden layer
  - indexLayer - number of hidden layer (0 - first layer, 1 - second layer, ....)
  - actFunction - activation function, which will be set
  - derivFunction - derivative of activation function, which will be set


- <b>setSoftmax(self,bool)</b>
  - bool - if true softmax function is activated on last layer (output layer) and its derivative, otherwise there's no function used


- <b>predict(self,inputData)</b> return prediction of network basing on given inputData. Of course input needs match to network structure.
  - inputData - set of input data


- <b>fit(self, inputData, expectedData, iterations=100, alpha=0.01, batchSize=-1, dropPercent=0.5, testInput=None, testExpected=None)</b>
    - inputData - set of training data
    - expectedData - set of expeceted ouput for training data
    - iterations - number of iterations for training
    - alpha - learning rate, number between 0 and 1
    - batchSize - size of batch, if no batch passed full batch is used
    - dropPercent - indicate percent of dropout, number between 0 and 1
    - testInput - set of testing data, used in the test for overfitting
    - testInput -  set of expeceted ouput for tested data, used in the test for overfitting

File <b>activationFunctions.py</b> contains few most popular activation functions and their derivative

To pass arrays, matrices and list of matrices use <b>np.array</b>

## Briefly about network
Network is build of layers, we can distinguish input, hidden and output layer.
Each layer is build of neurons, which is some kind of atom unit of networks. We need to adjust number of neurons in input layer to input data and similarly number of neurons in output layer is equal to number of expected output. For example in MNIST database we have images of digits 28x28 pixels, so our input is 784 pixels of image, and output is every possible digits 0-9. Structure of network could looks like that [784,hiddenLayer,10] then. Hidden layer is useful for more advanced techniques of deep learning. Without going into details I would say this network use backpropagation, gradient descent, batch mechanism and dropout. When we are preparing our network for training we should take into account
- learning rate (alpha), which says how fast network will learn
- iterations indicates how long network will learn
- leftRange and rightRange says about range of starting weights
- dropoutPercent is percent of neurons dropout, which helps us to avoid net overfitting
- activation functions are used to filter outputs of each layers

So us we can see there are few factors of network that should be adjust for our input data. It affects on training time and final accuracy. Sometimes it is really hard task to do. There are many unexplained terms here, so if interested in about details i can recommend <b>Grokking Deep Learning(Andrew W. Trask)</b> book, which is really great first step into neural networks.

## MNIST Test
Here i want to show some results for MNIST database
<table style="width:100%">
  <tr>
    <th>Structure</th>
    <th>Amount of training data</th>
    <th>Amount of testing data</th>
    <th>Iterations</th>
    <th>Alpha</th>
    <th>DropoutPercent</th>
    <th>Batch size</th>
    <th>Hidden activaton function</th>
    <th>Output activaton function</th>
    <th>Accuracy</th>
    <th>Training time</th>
  </tr>
  <tr>
    <td>[784,40,10]</td>
    <td>1000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.0005</td>
    <td>0.5</td>
    <td>Full</td>
    <td>ReLU</td>
    <td>Empty</td>
    <td>85%</td>
    <td>25s</td>
  </tr>
  <tr>
    <td>[784,40,10]</td>
    <td>10000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.00005</td>
    <td>0.5</td>
    <td>Full</td>
    <td>ReLU</td>
    <td>Empty</td>
    <td>89%</td>
    <td>242s</td>
  </tr>
  <tr>
    <td>[784,40,10]</td>
    <td>60000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.000005</td>
    <td>0.5</td>
    <td>Full</td>
    <td>ReLU</td>
    <td>Empty</td>
    <td>87%</td>
    <td>1332s</td>
  </tr>
  <tr>
    <td>[784,40,10]</td>
    <td>1000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.001</td>
    <td>0.5</td>
    <td>100</td>
    <td>ReLU</td>
    <td>Empty</td>
    <td>86%</td>
    <td>24s</td>
  </tr>
  <tr>
    <td>[784,40,10]</td>
    <td>10000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.001</td>
    <td>0.5</td>
    <td>100</td>
    <td>ReLU</td>
    <td>Empty</td>
    <td>91%</td>
    <td>260s</td>
  </tr>
  <tr>
    <td>[784,40,10]</td>
    <td>60000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.0001</td>
    <td>0.5</td>
    <td>100</td>
    <td>ReLU</td>
    <td>Empty</td>
    <td>92%</td>
    <td>1693s</td>
  </tr>
  <tr>
    <td>[784,100,10]</td>
    <td>1000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.2</td>
    <td>0.5</td>
    <td>100</td>
    <td>Tanh</td>
    <td>softmax</td>
    <td>86%</td>
    <td>24s</td>
  </tr>
  <tr>
    <td>[784,100,10]</td>
    <td>10000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.05</td>
    <td>0.5</td>
    <td>100</td>
    <td>Tanh</td>
    <td>softmax</td>
    <td>92%</td>
    <td>384s</td>
  </tr>
  <tr>
    <td>[784,100,10]</td>
    <td>60000</td>
    <td>10000</td>
    <td>350</td>
    <td>0.01</td>
    <td>0.5</td>
    <td>100</td>
    <td>Tanh</td>
    <td>softmax</td>
    <td>92%</td>
    <td>2239s</td>
  </tr>
</table>
