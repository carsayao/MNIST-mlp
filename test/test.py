import numpy as np
import os
import sys

# Number of inputs
INPUTS = 784
# Number of neurons
NEURONS = 10
# Number of examples to train
SAMPLES = 60000
# Number of test images
SAMPLES_T = 10000
# Number of epochs
EPOCHS = 3
# Learning Rate
LR = 0.001
# Array of epochs to store correct %
CORRECT = []

arg = sys.argv[1]
path = os.path.dirname(os.path.realpath(__file__))
test_images_dat = path + '/../MNIST/test_images.dat'

if arg == 'array':
    # newfp = np.memmap(test_images_dat, dtype='float32', shape=(SAMPLES_T,INPUTS))
    # newfp[:,1:] = newfp[:,1:]
    matrix = np.array([[1,2,3,4]
                      ,[5,6,7,8]])
    print(matrix) 
    # print(matrix.shape[0], matrix.shape[1])
    # matrix = np.c_[matrix, np.ones(2)]
    # print(matrix)
    # print(matrix.shape[0], matrix.shape[1])
    print('exp\n',np.exp(matrix))
    print('sum\n',np.sum(np.exp(matrix),axis=1))
    mul = np.sum(np.exp(matrix),axis=1)*np.ones((1,np.shape(matrix)[0]))
    print('mul\n',mul)
    print('tra\n',np.transpose(np.transpose(np.exp(matrix))/mul))
    matrix = 1 / (1 + np.exp(-matrix))
    print('mat\n',matrix)
    m = np.empty((10,2),dtype='float32')
    m.fill(.1)
    print('dim m: %s x %s' % (m.shape[0], m.shape[1]))
    print(m)