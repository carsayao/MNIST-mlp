import numpy as np
import os
import sys

# Number of inputs
INPUTS = 784
# Number of neurons
NEURONS = 10
# Number of hidden neurons
NEURONS_H = 10
# Number of examples to train
SAMPLES = 60000
# Number of test images
SAMPLES_T = 10000
# Number of epochs
# EPOCHS = int(sys.argv[2])
EPOCHS = 50
# Learning Rate
# LR = float(sys.argv[1])
LR = .01
# Array of epochs to store correct %
CORRECT = []
# Absolute path
path = os.path.dirname(os.path.realpath(__file__))
# Relative paths
train_images_dat = path + '/MNIST/train_images.dat'
train_labels_dat = path + '/MNIST/train_labels.dat'

# Initialize hidden and output neurons and weights
def init_neurons_weights():
    n_o = np.array(np.zeros((SAMPLES,NEURONS)))
    n_h = np.array(np.zeros((SAMPLES,NEURONS_H)))
    w_o = np.random.uniform(-.05,.05,size=(INPUTS+1,NEURONS))
    w_h = np.random.uniform(-.05,.05,size=(INPUTS+1,NEURONS_H))
    return n_o, n_h, w_o, w_h
# Load training images and labels from data and add bias to images
def load():
    x = np.memmap(train_images_dat, dtype='float32',
                  mode='r+', shape=(SAMPLES,INPUTS))
    y = np.memmap(train_labels_dat, dtype='float32',
                  mode='r+', shape=(SAMPLES,NEURONS))
    x = np.c_[x, np.ones(SAMPLES)]
    return x, y
def activate(n):
    n = 1 / (1 + np.exp(-n))
def forward(x, t, n_h, n_o, w_h, w_o):
    n_h = np.dot(x, w_h)
    print('dim n_h: %s x %s' % (n_h.shape[0], n_h.shape[1]))
    activate(n_h)
    print('dim a(n_h): %s x %s' % (n_h.shape[0], n_h.shape[1]))
    n_o = np.dot(x, w_o)
    print('dim n_o: %s x %s' % (n_o.shape[0], n_o.shape[1]))
    activate(n_o)
    print('dim a(n_o): %s x %s' % (n_o.shape[0], n_o.shape[1]))

    o_deltas = n_o * (1-n_o) * (t-n_o)
    print('dim o_deltas: %s x %s' % (o_deltas.shape[0], o_deltas.shape[1]))
    h_deltas = n_h * (1-n_h) * np.dot(w_o,np.transpose(o_deltas))
    print('dim h_deltas: %s x %s' % (h_deltas.shape[0], h_deltas.shape[1]))


def main():
    inputs, targets = load()
    oneurons, hneurons, oweights, hweights = init_neurons_weights()
    print('dim inputs: %s x %s' % (inputs.shape[0], inputs.shape[1]))
    print('dim targets: %s x %s' % (targets.shape[0], targets.shape[1]))
    print('dim oneurons: %s x %s' % (oneurons.shape[0], oneurons.shape[1]))
    print('dim hneurons: %s x %s' % (hneurons.shape[0], hneurons.shape[1]))
    print('dim oweights: %s x %s' % (oweights.shape[0], oweights.shape[1]))
    print('dim hweights: %s x %s' % (hweights.shape[0], hweights.shape[1]))
    forward(inputs, targets, hneurons, oneurons, hweights, oweights)

if __name__ == '__main__':
    main()
