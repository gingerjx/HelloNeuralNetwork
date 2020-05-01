import numpy as np
import time
import readData as rd
import neuralNetwork as nn
import activationFunctions as af

def mnist():
# -- Test results for full batch + ReLu + empty (ex1):
# -- [1] net [784,40,10], train 1000, test 10000, acurancy 85%, training time 25s, iteratins 350, alpha 0.0005 -- #
# -- [2] net [784,40,10], train 10000, test 10000, acurancy 89%, training time 242s, iteratins 350, alpha 0.00005 -- #
# -- [3] net [784,40,10], train 60000, test 10000, acurancy 87%, training time 1332s, iteratins 350, alpha 0.000005 -- #

# -- Test results for mini batch 100 + ReLu + empty (ex2):
# -- [4] net [784,40,10], train 1000, test 10000, acurancy 86%, training time 24s, iteratins 350, alpha 0.001 -- #
# -- [5] net [784,40,10], train 10000, test 10000, acurancy 91%, training time 260s, iteratins 350, alpha 0.001 -- #
# -- [6] net [784,40,10], train 60000, test 10000, acurancy 92%, training time 1693, iteratins 350, alpha 0.0001 -- #

# -- Test results for mini batch 100 + tanh + softmax (ex3):
# -- [7] net [784,100,10], train 1000, test 10000, acurancy 86%, training time 24s, iteratins 350, alpha 0.2 -- #
# -- [8] net [784,100,10], train 10000, test 10000, acurancy 92%, training time 384s, iteratins 350, alpha 0.05 -- #
# -- [9] net [784,100,10], train 60000, test 10000, acurancy 92%, training time 2239s, iteratins 350, alpha 0.01 -- #

# -- loading data -- #
    print("Load data...")
    trainLabels = rd.loadMnistLabels("data/train-labels-digits")
    trainImages = rd.loadMnistImages("data/train-images-digits")
    testLabels = rd.loadMnistLabels("data/t10k-labels-digits")
    testImages = rd.loadMnistImages("data/t10k-images-digits")
    print("...data loaded")
    print("trainLabels shape ", trainLabels.shape )
    print("trainImages shape ", trainImages.shape )
    print("testLabels shape ", testLabels.shape )
    print("testImages shape ", testImages.shape )

# -- initializing net -- #
    net = nn.DeepNetwork([784, 100, 10], leftRange=-0.01, rightRange=0.01)
    net.setHiddenAF(0, af.tanh, af.derivTanh)
    net.setSoftmax(True)

# -- training -- #
    start_time = time.time()
    net.fit(trainImages,trainLabels,batchSize=100, iterations=350, alpha=0.01, dropPercent=0.5, testInput=testImages,testExpected=testLabels)
    print("Training time ", time.time() - start_time, " seconds\n")

# -- testing -- #
    start_time = time.time()
    net.test(testImages,testLabels)
    print("Testing time ", time.time() - start_time, " seconds")

if __name__ == "__main__":
    mnist()

