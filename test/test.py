import numpy as np
import os
import sys

# Number of inputs
INPUTS = 784
# Number of neurons
NEURONS = 10
# Number of hidden neurons
NEURONS_H = 50
# Number of examples to train
SAMPLES = 60000
# Number of test images
SAMPLES_T = 10000
# Number of epochs
EPOCHS = 3
# Learning Rate
LR = 0.001
# Momentum term
# ALPHA = 0
# ALPHA = .9
ALPHA = .5
# Array of epochs to store correct %
CORRECT = []

arg = sys.argv[1]
path = os.path.dirname(os.path.realpath(__file__))
test_images_dat = path + '/../MNIST/test_images.dat'

if arg == 'array':
# if True:
    # newfp = np.memmap(test_images_dat, dtype='float32', shape=(SAMPLES_T,INPUTS))
    # newfp[:,1:] = newfp[:,1:]
    matrix = np.array([[1,2,3,4]
                      ,[5,6,7,8]])
    print(matrix) 
    print(matrix.shape[0], matrix.shape[1])
    sum = np.sum(matrix,axis=1)
    print('np.sum(matrix),axis=1 : %s' % (sum))
    # matrix = np.c_[matrix, np.ones(2)]
    # print(matrix)
    # print(matrix.shape[0], matrix.shape[1])

    # print('exp\n',np.exp(matrix))
    print('np.sum(np.exp(matrix),axis=1)\n',np.sum(np.exp(matrix),axis=1))
    # mul = np.sum(np.exp(matrix),axis=1)*np.ones((1,np.shape(matrix)[0]))
    # print('mul\n',mul)
    # print('tra\n',np.transpose(np.transpose(np.exp(matrix))/mul))
    # matrix = 1 / (1 + np.exp(-matrix))
    # print('mat\n',matrix)
    # m = np.empty((10,2),dtype='float32')
    # m.fill(.1)
    # print('dim m: %s x %s' % (m.shape[0], m.shape[1]))
    # print(m)

    # nread = SAMPLES
    # inputs = np.zeros((SAMPLES,INPUTS))
    # targets = np.zeros((1,SAMPLES))
    # tset = np.array((inputs,targets))
    # # print('tset:%sx%s' % (tset.shape[0],tset.shape[1]))
    # train_in = tset[0][:nread,:]
    # print('train_in:%sx%s' % (train_in.shape[0],train_in.shape[1]))
    # inputs = np.zeros((SAMPLES,10))
    # print('inputs:%sx%s' % (inputs.shape[0],inputs.shape[1]))
    # ndata = np.shape(inputs)[0]
    # negative_ones = -np.ones((ndata,1))
    # print('negative_ones: %sx%s' % (negative_ones.shape[0],negative_ones.shape[1]))
    # inputs = np.concatenate((inputs,negative_ones),axis=1)
    # print('inputs:%sx%s' % (inputs.shape[0],inputs.shape[1]))
    # x = np.zeros((SAMPLES,INPUTS+1))
    # print('x:%sx%s' % (x.shape[0],x.shape[1]))
    # print('y = np.c_[x, np.ones(SAMPLES)]')
    # y = np.c_[x, np.ones(SAMPLES)]
    # print('y:%sy%s' % (y.shape[0],y.shape[1]))
    # print('x:%sx%s' % (x.shape[0],x.shape[1]))
    
    # h_deltas = np.ones((SAMPLES,NEURONS_H+1))
    # h_deltas = np.array([[1,2,3,4],[5,6,7,8]])
    # print('\nh_deltas:%sx%s' % (h_deltas.shape[0],h_deltas.shape[1]))
    # print(h_deltas)
    # chop = h_deltas[:-1:]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')
    # chop = h_deltas[:1,:]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')
    # chop = h_deltas[:,1:]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')
    # chop = h_deltas[:,1:1]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')
    # chop = h_deltas[:,:1]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')
    # chop = h_deltas[:,:-1]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')

if arg == 'format':
    print('hello ' + f'{int(ALPHA*1000):03d}')
