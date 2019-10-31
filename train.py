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
    # inputs, samples, hidden units, learning rate, momentum, epochs
    def __init__(self, x, N, n_h, lr, alpha, epochs):
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
        try:
            n = 1 / (1 + np.exp(-n))
        except RuntimeWarning as r:
        # except:
            print('\nwarning! n==%s, r: %s' % (n,r))

    # Params: x==inputs, t==targets, n==neurons, w==weights, u==update,
    #         h==hidden, o==output
    def train(self, x, t):

        # Initialize prev update arrays
        u_h = np.zeros((np.shape(self.w_h)))
        u_o = np.zeros((np.shape(self.w_o)))

        # Add number of hidden units and momentum term to our statistics array
        self.CORRECT.append(self.NEURONS_H)
        self.CORRECT.append(self.ALPHA)

        # print("\nforward()")
        # print("w_o[0]\n",self.w_o[0])
        # print("w_h[0]\n",self.w_h[0])


        for e in range(self.EPOCHS):

            print('\nEpoch: %s lr: %s n: %s N: %s' % (e+1, self.LR, self.NEURONS_H, self.SAMPLES))

            # Get hidden units
            n_h = np.dot(x, self.w_h)
            # print("n_h = np.dot(x, self.w_h)\n", n_h)
            # Sigmoid
            self.activate(n_h)
            # Add bias
            n_h = np.c_[n_h,np.ones(self.SAMPLES)]
            # Get outputs
            n_o = np.dot(n_h, self.w_o)
            # Sigmoid
            self.activate(n_o)
            # print("n_o[0]\n", n_o[0])
            # Activate max of outputs
            for N in range(self.SAMPLES):
                n_o[N] = np.where(n_o[N]>=np.amax(n_o[N]),1,0)
            # Set target value t_k for output unit k to 0.9 if input is correct, 0.1 otherwise
            target_k = n_o * t
            for N in range(self.SAMPLES):
                target_k[N] = np.where(target_k[N]==1,0.9,0.1)
            # print("n_h[0]\n", n_h[0])
            # print("n_o[0]\n", n_o[0])

            # Output deltas
            # o_deltas = n_o * (1-n_o) * (t-n_o)
            o_deltas = n_o * (1-n_o) * (target_k-n_o)
            # print("n_o[0]\n", n_o[0])
            # print("t[0]\n", t[0])
            # print("target_k[0]\n", target_k[0])
            # print("o_deltas[0]\n", o_deltas[0])
            # Hidden deltas
            h_deltas = n_h * (1-n_h) * np.dot(o_deltas,np.transpose(self.w_o))
            # print('w_o[0]\n', self.w_o[0])
            # print("o_deltas[0]\n", o_deltas[0])
            # print("h_deltas[0]\n", h_deltas[0])

            # Input to hidden weight update
            # print("x[0]\n", x[0])
            # print("np.transpose(x)[0]\n", np.transpose(x)[0])
            # print("h_deltas[:,:-1]\n", h_deltas[:,:-1])
            # print("u_h:%sx%s" % (u_h.shape[0], u_h.shape[1]))
            # print("u_h[0]\n", u_h[0])
            u_h = self.LR * np.dot(np.transpose(x), h_deltas[:,:-1]) + self.ALPHA * u_h
            # print("after u_h:%sx%s" % (u_h.shape[0], u_h.shape[1]))
            # print("u_h[0]\n", u_h[0])
            # print("w_h[0]\n",self.w_h[0])
            self.w_h += u_h
            # print("w_h[0]\n",self.w_h[0])

            # Hidden to output weight update
            # print("u_o:%sx%s" % (u_o.shape[0], u_o.shape[1]))
            # print("u_o[0]\n", u_o[0])
            u_o = self.LR * np.dot(np.transpose(n_h), o_deltas) + self.ALPHA * u_o
            # print("after u_o:%sx%s" % (u_o.shape[0], u_o.shape[1]))
            # print("u_o[0]\n", u_o[0])
            self.w_o += u_o
            # print("w_o:%sx%s" % (self.w_o.shape[0], self.w_o.shape[1]))
            # print("w_o[0]\n",self.w_o[0])

            self.get_confusion(t, n_o)

        # hu == 'hidden units', m == 'momentum'
        save_stats0 = self.path + '/MNIST/train_hu' + f'{int(self.NEURONS_H):03d}'
        save_stats1 = '-m' + f'{int(self.ALPHA*1000):03d}'
        save_stats2 = '-ex' + str(self.SAMPLES)
        save_stats = save_stats0 + save_stats1 + save_stats2
        # save_stats = self.path+'/train_hu.csv'
        np.savetxt(fname=save_stats, X=self.CORRECT, delimiter=',')

            # Update confusion matrix
            # for N in range(self.SAMPLES):
            #     self.n_o[N] = np.where(self.n_o[N]>=np.amax(self.n_o[N]),1,0)
            # temp_target = np.zeros((self.SAMPLES,self.NEURONS))
            # for N in range(self.SAMPLES):
            #     temp_target[N] = np.where(self.t[N]>=np.amax(self.t[N]),1,0)

    # def backward(self, x, t, u_h, u_o):
        # Get deltas

    def get_confusion(self, t, n_o):
        # Update confusion matrix
        # for N in range(self.SAMPLES):
            # n_o[N] = np.where(n_o[N]>=np.amax(n_o[N]),1,0)

        # temp_target = np.zeros((self.SAMPLES,self.NEURONS))
        # for N in range(self.SAMPLES):
        #     temp_target[N] = np.where(t[N]>=np.amax(t[N]),1,0)

        confusion = np.dot(np.transpose(t), n_o)
        accuracy = np.trace(confusion) / self.SAMPLES
        self.CORRECT.append(accuracy)
        # print("confusion",confusion)
        print('  %:', round(accuracy*100, 4))

def main():
    # inputs, samples, hidden units, learning rate, momentum, epochs
    run = mlp(784, 60000, 50, 0.1, 0.9, 50)
    inputs, targets = run.load()
    run.train(inputs, targets)
    # run = mlp()
    # run.forward()
    # run.back

if __name__ == '__main__':
    main()
