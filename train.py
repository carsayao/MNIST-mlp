from mlxtend.data import loadlocal_mnist
import numpy as np
import os
import sys
import warnings

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
    # EPOCHS = int(sys.argv[2])
    EPOCHS = 5
    # Learning Rate
    # LR = float(sys.argv[1])
    LR = .01
    # Momentum term
    ALPHA = 0
    # Array of epochs to store correct %
    CORRECT = []
    # Absolute path
    path = os.path.dirname(os.path.realpath(__file__))
    # Relative paths
    train_images_dat = path + '/MNIST/train_images.dat'
    train_labels_dat = path + '/MNIST/train_labels.dat'

    def __init__(self):
        self.x, self.t = self.load()
        self.n_o, self.n_h, self.w_o, self.w_h = self.init_neurons_weights()

    # Initialize hidden and output neurons and weights
    def init_neurons_weights(self):
        n_o = np.array(np.zeros((self.SAMPLES,self.NEURONS)))
        n_h = np.array(np.zeros((self.SAMPLES,self.NEURONS_H+1)))
        w_o = np.random.uniform(-.05,.05,size=(self.NEURONS_H+1,self.NEURONS))
        w_h = np.random.uniform(-.05,.05,size=(self.INPUTS+1,self.NEURONS_H))
        print("w_o[0]\n",w_o[0])
        print("w_h[0]\n",w_h[0])
        return n_o, n_h, w_o, w_h

    def read_train_data(self):
        train_images_raw = self.path + '/MNIST/rawdata/train-images.idx3-ubyte'
        train_labels_raw = self.path + '/MNIST/rawdata/train-labels.idx1-ubyte'
        x, y = loadlocal_mnist(
                images_path = train_images_raw,
                labels_path = train_labels_raw)
        return x, y

    # Load training images and labels from data and add bias to images
    def load(self):
        x = np.memmap(self.train_images_dat, dtype='float64',
                    mode='r+', shape=(self.SAMPLES,self.INPUTS))
        t = np.memmap(self.train_labels_dat, dtype='float64',
                    mode='r+', shape=(self.SAMPLES,self.NEURONS))
        x = np.c_[x, np.ones(self.SAMPLES)]
        return x, t
    
    # Sigmoid function
    def activate(self, n):
        try:
            n = 1 / (1 + np.exp(-n))
        except RuntimeWarning as r:
        # except:
            print('\nwarning! n==%s, r: %s' % (n,r))

    # Params: x==inputs, t==targets, n==neurons, w==weights, u==update,
    #         h==hidden, o==output
    def forward(self):

        # Initialize prev update arrays
        u_h = np.zeros(np.shape(self.w_h))
        u_o = np.zeros(np.shape(self.w_o))

        # Add number of hidden units and momentum term to our statistics array
        self.CORRECT.append(self.NEURONS_H)
        self.CORRECT.append(self.ALPHA)

        print("\nforward()")
        print("w_o[0]\n",self.w_o[0])
        print("w_h[0]\n",self.w_h[0])

        for e in range(self.EPOCHS):

            confusion = []
            print('\nEpoch: %s lr: %s n: %s N: %s' % (e+1, self.LR, self.NEURONS_H, self.SAMPLES))

            # Get hidden units, activate, add bias, then 
            self.n_h = np.dot(self.x, self.w_h)
            self.activate(self.n_h)
            self.n_h = np.c_[self.n_h,np.ones(self.SAMPLES)]
            self.n_o = np.dot(self.n_h, self.w_o)
            self.activate(self.n_o)

            # Update confusion matrix
            # for N in range(self.SAMPLES):
            #     self.n_o[N] = np.where(self.n_o[N]>=np.amax(self.n_o[N]),1,0)
            # temp_target = np.zeros((self.SAMPLES,self.NEURONS))
            # for N in range(self.SAMPLES):
            #     temp_target[N] = np.where(self.t[N]>=np.amax(self.t[N]),1,0)

            # Get deltas
            o_deltas = self.n_o * (1-self.n_o) * (self.t-self.n_o)
            print("n_o[0]\n", self.n_o[0])
            print("t[0]\n", self.t[0])
            h_deltas = self.n_h * (1-self.n_h) * np.dot(o_deltas,np.transpose(self.w_o))
            print("o_deltas[0]\n", o_deltas[0])

            # Hidden to output weight update
            u_o = self.LR * np.dot(np.transpose(self.n_h), o_deltas) + self.ALPHA * u_o
            self.w_o -= u_o
            print("w_o[0]\n",self.w_o[0])

            # Input to hidden weight update
            u_h = self.LR * np.dot(np.transpose(self.x), h_deltas[:,:-1]) + self.ALPHA * u_h
            self.w_h -= u_h
            print("w_h[0]\n",self.w_h[0])


            # Update confusion matrix
            for N in range(self.SAMPLES):
                self.n_o[N] = np.where(self.n_o[N]>=np.amax(self.n_o[N]),1,0)
            temp_target = np.zeros((self.SAMPLES,self.NEURONS))
            for N in range(self.SAMPLES):
                temp_target[N] = np.where(self.t[N]>=np.amax(self.t[N]),1,0)

            confusion = np.dot(np.transpose(self.t), self.n_o)
            accuracy = np.trace(confusion) / self.SAMPLES
            self.CORRECT.append(accuracy)
            print('  %:', round(accuracy*100, 4))

        # hu == 'hidden units', m == 'momentum'
        save_stats0 = self.path + '/MNIST/train_hu' + f'{int(self.NEURONS_H):03d}'
        save_stats1 = '-m' + f'{int(self.ALPHA*1000):03d}'
        save_stats2 = '-ex' + str(self.SAMPLES)
        save_stats = save_stats0 + save_stats1 + save_stats2
        # save_stats = self.path+'/train_hu.csv'
        np.savetxt(fname=save_stats, X=self.CORRECT, delimiter=',')

def main():
    run = mlp()
    run.forward()
    # run.back

if __name__ == '__main__':
    main()
