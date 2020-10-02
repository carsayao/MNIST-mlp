from mlxtend.data import loadlocal_mnist
import numpy as np
import os
import sys
FLOAT=True

# Number of inputs
INPUTS = 784
# Number of neurons
NEURONS = 10
# Number of examples to train
SAMPLES = 60000
# Number of test images
SAMPLES_T = 10000
# Array of epochs to store correct %
CORRECT = []
# Absolute path
#path = os.path.dirname(os.path.realpath(__file__))
path = './MNIST/rawdata/'
# Relative paths
print(path)
train_images_raw = path + 'train-images.idx3-ubyte'
train_labels_raw = path + 'train-labels.idx1-ubyte'
test_images_raw  = path + 't10k-images.idx3-ubyte'
test_labels_raw  = path + 't10k-labels.idx1-ubyte'
train_images_dat = path + '/../memmap/train_images.dat'
train_labels_dat = path + '/../memmap/train_labels.dat'
test_images_dat  = path + '/../memmap/test_images.dat'
test_labels_dat  = path + '/../memmap/test_labels.dat'

def read_to_dat():
    # Load raw images from ubytes
    def read_train_data():
        x, y = loadlocal_mnist(
                images_path = train_images_raw,
                labels_path = train_labels_raw)
        return x, y
    def read_test_data():
        x, y = loadlocal_mnist(
                images_path = test_images_raw,
                labels_path = test_labels_raw)
        return x, y
    def convert_targets_train(train_array):
        target_array = np.zeros((SAMPLES, NEURONS), dtype="float32")
        for t in range(SAMPLES):
            target_array[t][int(train_array[t])] = 1
        return target_array
    def convert_targets_test(test_array):
        target_array = np.zeros((SAMPLES_T, NEURONS), dtype='float32')
        for t in range(SAMPLES_T):
            target_array[t][int(test_array[t])] = 1
        return target_array

    train_images, train_labels = read_train_data()
    train_labels = convert_targets_train(train_labels)
    # Normalize
    train_images = train_images / 255
    # Create memmap pointer on disk and read into .dats
    fp0 = np.memmap(train_images_dat, dtype='float64',
                    mode='w+', shape=(SAMPLES,INPUTS))
    # fp0 = np.memmap(train_images_dat, mode='w+', shape=(SAMPLES,INPUTS))
    # Copy into file pointer
    fp0[:] = train_images[:]
    del fp0
    fp1 = np.memmap(train_labels_dat, dtype='float64',
                    mode='w+', shape=(SAMPLES,NEURONS))
    fp1[:] = train_labels[:]
    del fp1

    # Same as above but with test samples
    test_images, test_labels = read_test_data()
    test_labels = convert_targets_test(test_labels)
    test_images = test_images / 255
    fp2 = np.memmap(test_images_dat, dtype='float64',
                    mode='w+', shape=(SAMPLES_T,INPUTS))
    fp2[:] = test_images[:]
    del fp2
    fp3 = np.memmap(test_labels_dat, dtype='float64',
                    mode='w+', shape=(SAMPLES_T,NEURONS))
    fp3[:] = test_labels[:]
    del fp3

read_to_dat()
