import numpy as np
import time
from loadMNIST import load_mnist_labels, load_mnist_images
from neuralNetwork import Neural

def network_train():
    """ After code refactoring training executes 6 times faster!!
        And after parameters adjustment accuracy are on the similar level.
        Mainly, implementation of activation function changed (it is source of this acceleration)
        and API, which is now more programmer-friendly and scalable. """
    print("Loading data...")
    train_labels = load_mnist_labels("data/train-labels-digits")
    train_images = load_mnist_images("data/train-images-digits")
    test_labels = load_mnist_labels("data/t10k-labels-digits")
    test_images = load_mnist_images("data/t10k-images-digits")
    print("...data loaded")

    model = Neural()
    model.add_input(784)
    model.add_fully_connected(16, 'relu')
    model.add_fully_connected(32, 'relu')
    model.add_fully_connected(10, 'linear')

    start_time = time.time()
    model.fit(train_images, train_labels, iterations=500, alpha=1.0, drop_percent=0.5, test_input=test_images, test_expected=test_labels)
    print("Training time ", time.time() - start_time, " seconds\n")

    model.save_network("./network")

def network_test():
    print("Loading data...")
    test_labels = load_mnist_labels("data/t10k-labels-digits")
    test_images = load_mnist_images("data/t10k-images-digits")
    print("...data loaded")

    model = Neural()
    model.load_network("./network")

    prediction = model.predict(test_images)
    correct = 0
    error = 0
    for i in range(len(prediction)):
        net_answer = np.argmax(prediction[i])
        expected_answer = np.argmax(test_labels[i])
        correct += (net_answer == expected_answer)
        error += (net_answer - expected_answer)**2
    error /= test_images.shape[0]*test_images.shape[1]
    print("Accuracy: %.2f  error %.6f" % (((correct/len(prediction))*100), error))

if __name__ == "__main__":
    #network_train()
    network_test()



"""     -----------------------------------     Results     ------------------------------------ 
                            Below you can find some results of this network     
   Accuracy  Training Time      Neurons         Functions      Train Data  Test Data  Iters  Alpha  Dropout 
 -----------    ------       ------------    ---------------  ------------ ---------  -----  ----- ---------   
    89.96%       250s          784-40-10      in-relu-linear     60000      10000      350    0.7    0.5
    90.66%       480s          784-100-10     in-relu-linear     60000      10000      350    0.7    0.5       
    90.48%       277s         784-16-32-10  in-relu-relu-linear  60000      10000      350    1.0    0.5
    86.42%       285s         784-16-32-10  in-tanh-tanh-linear  60000      10000      350    1.0    0.5
    81.96%       393s          784-40-10      in-tanh-linear     60000      10000      500    0.1    0.5
    91.46%       354s          784-40-10      in-relu-linear     60000      10000      500    1.0    0.5
    91.77%       986s          784-100-10     in-relu-linear     60000      10000      750    1.0    0.5
    56.35%       82s           784-100-10     in-tanh-softmax    1000       10000      3500   1.0    0.5
"""

