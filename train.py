from mlxtend.data import loadlocal_mnist
import numpy as np
from scipy.special import expit
import os
import sys
import warnings
warnings.filterwarnings("error")
# DEBUG = True
DEBUG = False
np.set_printoptions(threshold=sys.maxsize)

# t:	    ( SAMPLES , NEURONS )
# inputs:	( SAMPLES , INPUTS+1 )
# w_h:	    ( INPUTS+1 , HIDDEN )
# n_h:	    ( SAMPLES , HIDDEN+1 )
# w_o:	    ( HIDDEN+1 , NEURONS )
# n_o:	    ( SAMPLES , NEURONS )
# o_delt:	( SAMPLES, NEURONS )
# h_delt:	( SAMPLES , HIDDEN+1 )
# u_o:	    ( HIDDEN+1 , NEURONS )
# w_o:	    ( HIDDEN+1 , NEURONS )
# u_h:	    ( INPUTS+1 , HIDDEN )
# w_h:	    ( INPUTS+1 , HIDDEN )

class mlp:
    # inputs, samples, hidden units, learning rate, momentum, weight decay, epochs
    def __init__(self, inputs, N, hidden, lr, mom, decay, epochs):
        self.INPUTS = inputs
        self.NEURONS = 10
        self.HIDDEN = hidden
        self.SAMPLES = N
        self.EPOCHS = epochs
        self.LR = lr
        self.ALPHA = mom
        self.LAMBDA = decay
        # Array of epochs to store correct %
        self.CORRECT = []
        # Absolute path
        self.path = os.path.dirname(os.path.realpath(__file__))
        # Relative paths
        self.train_images_dat = self.path + '/MNIST/train_images.dat'
        self.train_labels_dat = self.path + '/MNIST/train_labels.dat'

        self.w_o = np.random.uniform(-0.05,0.05,size=(self.HIDDEN+1,self.NEURONS))
        self.w_h = np.random.uniform(-0.05,0.05,size=(self.INPUTS+1,self.HIDDEN))

        # self.w_o = np.random.randint(low=-50,high=50,size=(self.HIDDEN+1,self.NEURONS)) / 1000
        # self.w_h = np.random.randint(low=-50,high=50,size=(self.INPUTS+1,self.HIDDEN)) / 1000

    # Initialize hidden and output neurons and weights
    def init_neurons_weights(self):
        output = np.array(np.zeros((self.SAMPLES,self.NEURONS)))
        hidden = np.array(np.zeros((self.SAMPLES,self.HIDDEN+1)))
        oweights = np.random.uniform(-0.05,0.05,size=(self.HIDDEN+1,self.NEURONS))
        hweights = np.random.uniform(-0.05,0.05,size=(self.INPUTS+1,self.HIDDEN))
        return output, hidden, oweights, hweights

    def read_train_data(self):
        train_images_raw = self.path + '/MNIST/rawdata/train-images.idx3-ubyte'
        train_labels_raw = self.path + '/MNIST/rawdata/train-labels.idx1-ubyte'
        x, y = loadlocal_mnist(
                images_path = train_images_raw,
                labels_path = train_labels_raw)
        x = x / 255
        return x[:self.SAMPLES], y[:self.SAMPLES]

    # Load training images and labels from data and add bias to images
    def load(self):
        load_inputs = np.memmap(self.train_images_dat, dtype='float64',
                    mode='r+', shape=(self.SAMPLES,self.INPUTS))
        load_targets = np.memmap(self.train_labels_dat, dtype='float64',
                    mode='r+', shape=(self.SAMPLES,self.NEURONS))
        load_inputs = np.c_[load_inputs, np.ones(self.SAMPLES)]
        return load_inputs, load_targets
    
    # Sigmoid function
    def activate(self, n):
        return expit(n)

    def shuffle_sets(self, samples, targets):
        rng_state = np.random.get_state()
        np.random.shuffle(samples)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

    # Params: x==inputs, t==targets, n==neurons, w==weights, u==update,
    #         h==hidden, o==output
    def train(self, x, t):
        # Add bias
        inputs = np.c_[x, np.ones(self.SAMPLES)]
        # Convert to target array
        target_array = np.zeros((self.SAMPLES, self.NEURONS))
        for i in range(self.SAMPLES):
            target_array[i][int(t[i])] = 1

        # Initialize prev update arrays
        u_h = np.zeros((np.shape(self.w_h)))
        u_o = np.zeros((np.shape(self.w_o)))

        # Add number of hidden units and momentum term to our statistics array
        self.CORRECT.append(self.HIDDEN)
        self.CORRECT.append(self.ALPHA)

        for e in range(self.EPOCHS):
            self.shuffle_sets(inputs, target_array)
            confusion = np.zeros((10,10))
            print('\n\n\n\nEpoch: %s lr: %s n: %s N: %s decay: %s'
                   % (e+1, self.LR, self.HIDDEN, self.SAMPLES, self.LAMBDA))

            # Get hidden units
            n_h = np.dot(inputs, self.w_h)
            # Sigmoid
            n_h = self.activate(n_h)
            # Add bias
            n_h = np.c_[n_h,np.ones(self.SAMPLES)]
            # Get outputs
            n_o = np.dot(n_h, self.w_o)

            # 1 Activate max of outputs by storing activations of n_o in predictions array.
            #     * predictions array is used to compute confusion matrix.
            # 2 Activate on output neurons.
            # 3 Find the max of each sample in predictions and give it a 1 there, 0 otherwise.
            # 4 Turn target array into array of 0.9s and 0.1s.
            #     * for calculating deltas.
            predictions = np.empty((np.shape(n_o)))
            predictions = self.activate(n_o)
            n_o = self.activate(n_o)
            for N in range(self.SAMPLES):
                predictions[N] = np.where(predictions[N]>=np.amax(predictions[N]),1,0)
                # predictions[N] = np.where(predictions[N]>=np.amax(n_o[N]),1,0)
            # Set target value t_k for output unit k to 0.9 if input is correct, 0.1 otherwise
            target_k = predictions * target_array
            for N in range(self.SAMPLES):
                target_k[N] = np.where(target_array[N]==1,0.9,0.1)

            # Backprop

            # Output deltas
            o_deltas = np.empty((self.SAMPLES, self.NEURONS))
            o_deltas = 1 * n_o * (1-n_o) * (target_k-n_o)
            # Hidden deltas
            h_deltas = np.empty((self.SAMPLES, self.HIDDEN+1))
            h_deltas = 1 * n_h * (1-n_h) * np.dot(o_deltas,np.transpose(self.w_o))
            # Input to hidden weight update
            u_h = self.LR * np.dot(np.transpose(inputs), h_deltas[:,:-1]) + self.ALPHA * u_h - self.LAMBDA * u_h
            self.w_h += u_h
            # Hidden to output weight update
            u_o = self.LR * np.dot(np.transpose(n_h), o_deltas) + self.ALPHA * u_o
            self.w_o += u_o

            self.get_confusion(confusion, target_array, predictions)

        # Save the stats of each epoch. First element in array number of hidden units
        # and the second momentum. hu == 'hidden units', m == 'momentum'
        save_stats = (self.path + '/MNIST/train_hu' + f'{int(self.HIDDEN):03d}'
                      + '-m' + f'{int(self.ALPHA*1000):03d}'
                      + '-ex' + str(self.SAMPLES) + '.csv')
        np.savetxt(fname=save_stats, X=self.CORRECT, delimiter=',')
        print("Saved file as %s" % save_stats)

    # def forward(self):
    # def backward(self):

    # Write to confusion matrix
    def get_confusion(self, conf, target, pred):
        conf = np.dot(np.transpose(target), pred)
        accuracy = np.trace(conf) / self.SAMPLES
        self.CORRECT.append(accuracy)
        print("\nconf")
        print(conf)
        print('  %:', round(accuracy*100, 4))

def main():
    # inputs, samples, hidden units, learning rate, momentum, decay, epochs
    run = mlp(784, 60000, 10, 0.1, 0.9, .2, 6)
    inputs, targets = run.read_train_data()
    run.train(inputs, targets)

if __name__ == '__main__':
    main()
