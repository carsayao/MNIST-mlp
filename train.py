from mlxtend.data import loadlocal_mnist
import numpy as np
from scipy.special import expit
import os
import sys
import warnings
warnings.filterwarnings("error")
# DEBUG = True
DEBUG = False
# np.set_printoptions(threshold=sys.maxsize)

# t:	    ( SAMPLES , 10 )
# x:	    ( SAMPLES , INPUTS+1 )
# w_h:	( INPUTS+1 , NEURONS_H )
# n_h:	( SAMPLES , NEURONS_H+1 )
# w_o:	( NEURONS_H+1 , NEURONS )
# n_o:	( SAMPLES , NEURONS )
# o_delt:	( SAMPLES, NEURONS )
# h_delt:	( SAMPLES , NEURONS_H+1 )
# u_o:	( NEURONS_H+1 , NEURONS )
# w_o:	( NEURONS_H+1 , NEURONS )
# u_h:	( INPUTS+1 , NEURONS_H )
# w_h:	( INPUTS+1 , NEURONS_H )

class mlp:
    # inputs, samples, hidden units, learning rate, momentum, weight decay, epochs
    def __init__(self, x, N, n_h, lr, alpha, decay, epochs):
        # Number of inputs
        self.INPUTS = x
        # Number of neurons
        self.NEURONS = 10
        # Number of hidden neurons
        self.NEURONS_H = n_h
        # Number of examples to train
        self.SAMPLES = N
        # Number of test images
        self.SAMPLES_T = 10000
        # Number of epochs
        # EPOCHS = int(sys.argv[2])
        self.EPOCHS = epochs
        # Learning Rate
        # LR = float(sys.argv[1])
        self.LR = lr
        # Momentum term
        self.ALPHA = alpha
        self.LAMBDA = decay
        # Array of epochs to store correct %
        self.CORRECT = []
        # Absolute path
        self.path = os.path.dirname(os.path.realpath(__file__))
        # Relative paths
        self.train_images_dat = self.path + '/MNIST/train_images.dat'
        self.train_labels_dat = self.path + '/MNIST/train_labels.dat'
        # self.w_o = np.random.uniform(-0.05,0.05,size=(self.NEURONS_H+1,self.NEURONS))
        # self.w_h = np.random.uniform(-0.05,0.05,size=(self.INPUTS+1,self.NEURONS_H))

        self.w_o = np.random.randint(low=-50,high=50,size=(self.NEURONS_H+1,self.NEURONS)) / 1000
        self.w_h = np.random.randint(low=-50,high=50,size=(self.INPUTS+1,self.NEURONS_H)) / 1000

        # self.x, self.t = self.load()
        # self.n_o, self.n_h, self.w_o, self.w_h = self.init_neurons_weights()
        # self.x
        # self.t
        # self.n_o
        # self.n_h
        # self.w_o
        # self.w_h

    # Initialize hidden and output neurons and weights
    def init_neurons_weights(self):
        output = np.array(np.zeros((self.SAMPLES,self.NEURONS)))
        hidden = np.array(np.zeros((self.SAMPLES,self.NEURONS_H+1)))
        oweights = np.random.uniform(-0.05,0.05,size=(self.NEURONS_H+1,self.NEURONS))
        hweights = np.random.uniform(-0.05,0.05,size=(self.INPUTS+1,self.NEURONS_H))
        # print("w_o[0]\n",w_o[0])
        # print("w_h[0]\n",w_h[0])
        return output, hidden, oweights, hweights

    def read_train_data(self):
        train_images_raw = self.path + '/MNIST/rawdata/train-images.idx3-ubyte'
        train_labels_raw = self.path + '/MNIST/rawdata/train-labels.idx1-ubyte'
        x, y = loadlocal_mnist(
                images_path = train_images_raw,
                labels_path = train_labels_raw)
        return x, y

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
        # try:
        #     return 1 / (1 + np.exp(-n))
        # except RuntimeWarning:
        #     print('\nwarning! n==%s' % n)
        # return 1/(1+expit(-1*n))
        return expit(n)

    def shuffle_sets(self, samples, targets):
        rng_state = np.random.get_state()
        np.random.shuffle(samples)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

    # Params: x==inputs, t==targets, n==neurons, w==weights, u==update,
    #         h==hidden, o==output
    def train(self, x, t):

        # Initialize prev update arrays
        u_h = np.zeros((np.shape(self.w_h)))
        u_o = np.zeros((np.shape(self.w_o)))

        # Add number of hidden units and momentum term to our statistics array
        self.CORRECT.append(self.NEURONS_H)
        self.CORRECT.append(self.ALPHA)

        for e in range(self.EPOCHS):
            self.shuffle_sets(x, t)
            confusion = np.zeros((10,10))
            print('\n\n\n\nEpoch: %s lr: %s n: %s N: %s' % (e+1, self.LR, self.NEURONS_H, self.SAMPLES))
            if DEBUG:
                print("\nu_h")
                print(u_h[0])
                print("\nu_o[0]")
                print(u_o[0])
                print("\nw_h[0]")
                print(self.w_h[0])
                print("\nw_o[0]")
                print(self.w_o[0])
            # Get hidden units
            n_h = np.dot(x, self.w_h)
            # print("n_h = np.dot(x, self.w_h)\n", n_h)
            # Sigmoid
            n_h = self.activate(n_h)
            # Add bias
            n_h = np.c_[n_h,np.ones(self.SAMPLES)]
            # Get outputs
            n_o = np.dot(n_h, self.w_o)
            if DEBUG:
                print("\n\n\tActivations")
                print("\nn_h[0]")
                print(n_h[0])
                print("\nn_o[0]")
                print(n_o[0])
                print("\nt[0]")
                print(t[0])

            # 1 Activate max of outputs by storing activations of n_o in predictions array.
            #     * predictions array is used to compute confusion matrix.
            # 2 Activate on output neurons.
            # 3 Find the max of each sample in predictions and give it a 1 there, 0 otherwise.
            # 4 Turn target array into array of 0.9s and 0.1s.
            #     * forcalculating deltas.
            # 
            predictions = np.empty((np.shape(n_o)),dtype="float32")
            predictions = self.activate(n_o)
            n_o = self.activate(n_o)
            for N in range(self.SAMPLES):
                # predictions[N] = np.where(predictions[N]>=np.amax(predictions[N]),1,0)
                predictions[N] = np.where(predictions[N]>=np.amax(predictions[N]),1,0)
            # Set target value t_k for output unit k to 0.9 if input is correct, 0.1 otherwise
            target_k = predictions * t
            for N in range(self.SAMPLES):
                target_k[N] = np.where(t[N]==1,0.9,0.1)

            # Backprop

            if DEBUG:
                print("\npredictions[0:10]")
                print(predictions[0:10])
                print("\ntarget_k[0:10]")
                print(target_k[0:10])
                print("\n\nn_o[0]")
                print(n_o[0])
            # Output deltas
            o_deltas = np.empty((self.SAMPLES, self.NEURONS))
            o_deltas = n_o * (1-n_o) * (target_k-n_o)
            if DEBUG:
                print("\no_deltas = n_o * (1-n_o) * (target_k-n_o)")
                print("\no_deltas[0]")
                print(o_deltas[0])
                print("\n\nw_o[0]")
                print(self.w_o[0])
                print("\nn_h[0]")
                print(n_h[0])
            # Hidden deltas
            h_deltas = np.empty((self.SAMPLES, self.NEURONS_H+1))
            h_deltas = n_h * (1-n_h) * np.dot(o_deltas,np.transpose(self.w_o))
            if DEBUG:
                print("\nh_deltas = n_h * (1-n_h) * np.dot(o_deltas,np.transpose(self.w_o))")
                print("\nh_deltas")
                print(h_deltas)
                print("\nu_h[-2:]")
                print(u_h[-2:])
            # Input to hidden weight update
            u_h = self.LR * np.dot(np.transpose(x), h_deltas[:,:-1]) + self.ALPHA * u_h - self.LAMBDA * u_h
            if DEBUG:
                print("\nu_h = self.LR * np.dot(np.transpose(x), h_deltas[:,:-1]) + self.ALPHA * u_h")
                print("\nu_h:%sx%s" % u_h.shape)
                print("\nu_h[-2:]")
                print(u_h[-2:])
                print("\nw_h[0]")
                print(self.w_h[0])
            self.w_h += u_h
            if DEBUG:
                print("\nself.w_h += u_h")
                print("\nw_h[0]")
                print(self.w_h[0])
                print("\nn_h[0]")
                print(n_h[0])
                print("\no_deltas")
                print(o_deltas[0])
                print("\nu_o")
                print(u_o[0])
            # Hidden to output weight update
            u_o = self.LR * np.dot(np.transpose(n_h), o_deltas) + self.ALPHA * u_o
            if DEBUG:
                print("\nu_o = self.LR * np.dot(np.transpose(n_h), o_deltas) + self.ALPHA * u_o")
                print("\nu_o[0]")
                print(u_o[0])
                print("\nw_o[0]")
                print(self.w_o[0])
            self.w_o += u_o
            if DEBUG:
                print("\nself.w_o += u_o")
                print("\nw_o[0]")
                print(self.w_o)

            self.get_confusion(confusion, t, predictions)

        # Save the stats of each epoch. First element in array number of hidden units
        # and the second momentum. hu == 'hidden units', m == 'momentum'
        #
        save_stats = (self.path + '/MNIST/train_hu' + f'{int(self.NEURONS_H):03d}'
                      + '-m' + f'{int(self.ALPHA*1000):03d}'
                      + '-ex' + str(self.SAMPLES) + '.csv')
        np.savetxt(fname=save_stats, X=self.CORRECT, delimiter=',')
        print("Saved file as %s" % save_stats)

    # Write to confusion matrix
    def get_confusion(self, conf, t, pred):
        if DEBUG:
            print("\nt")
            print(t)
            print("\npred")
            print(pred)
        conf = np.dot(np.transpose(t), pred)
        accuracy = np.trace(conf) / self.SAMPLES
        self.CORRECT.append(accuracy)
        print("\nconf")
        print(conf)
        print('  %:', round(accuracy*100, 4))

def main():
    # inputs, samples, hidden units, learning rate, momentum, decay, epochs
    run = mlp(784, 60000, 10, 0.1, 0.9, .2, 6)
    inputs, targets = run.load()
    run.train(inputs, targets)

if __name__ == '__main__':
    main()
