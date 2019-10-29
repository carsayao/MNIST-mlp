import numpy as np
import os
import sys
import warnings

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
    LR = .001
    # Momentum term
    ALPHA = .9
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
        return n_o, n_h, w_o, w_h

    # Load training images and labels from data and add bias to images
    def load(self):
        x = np.memmap(self.train_images_dat, dtype='float64',
                    mode='r+', shape=(self.SAMPLES,self.INPUTS))
        t = np.memmap(self.train_labels_dat, dtype='float64',
                    mode='r+', shape=(self.SAMPLES,self.NEURONS))
        x = np.c_[x, np.ones(self.SAMPLES)]
        return x, t

    def activate(self, n):
        try:
            n = 1 / (1 + np.exp(-n))
        except RuntimeWarning as r:
            print('\nwarning! n==%s, r: %s' (n,r))

    # Params: x==inputs, t==targets, n==neurons, w==weights, u==update,
    #         h==hidden, o==output
    def forward(self):
        u_h = np.zeros(np.shape(self.w_h))
        u_o = np.zeros(np.shape(self.w_o))
        # print('\nt: %s x %s' % (self.t.shape[0], self.t.shape[1]))
        # print('x: %s x %s' % (self.x.shape[0], self.x.shape[1]))
        # print('w_h: %s x %s' % (self.w_h.shape[0], self.w_h.shape[1]))
        # print('n_h: %s x %s' % (self.n_h.shape[0], self.n_h.shape[1]))
        # print('w_o: %s x %s' % (self.w_o.shape[0], self.w_o.shape[1]))
        # print('n_o: %s x %s' % (self.n_o.shape[0], self.n_o.shape[1]))
        self.CORRECT.append(self.NEURONS_H)
        self.CORRECT.append(self.ALPHA)
        for e in range(self.EPOCHS):
            confusion = []
            print('Epoch: %s lr: %s n: %s N: %s' % (e+1, self.LR, self.NEURONS_H, self.SAMPLES))
            self.n_h = np.dot(self.x, self.w_h)
            # print('n_h: %s x %s' % (self.n_h.shape[0], self.n_h.shape[1]))
            self.n_h = np.c_[self.n_h,np.ones(self.SAMPLES)]
            # print('n_h = np.dot(x,w_h): %s x %s' % (self.n_h.shape[0], self.n_h.shape[1]))
            self.activate(self.n_h)
#            # print('a(n_h): %s x %s' % (n_h.shape[0], n_h.shape[1]))
            self.n_o = np.dot(self.n_h, self.w_o)
            # print('n_o = np.dot(n_h, w_o): %s x %s' % (self.n_o.shape[0], self.n_o.shape[1]))
#            # print('n_o: %s x %s' % (n_o.shape[0], n_o.shape[1]))
            self.activate(self.n_o)
#            # print('a(n_o): %s x %s' % (n_o.shape[0], n_o.shape[1]))

            o_deltas = self.n_o * (1-self.n_o) * (self.t-self.n_o)
            # print('o_deltas: %s x %s' % (o_deltas.shape[0], o_deltas.shape[1]))

            h_deltas = self.n_h * (1-self.n_h) * np.dot(o_deltas,np.transpose(self.w_o))
            # print('h_deltas: %s x %s' % (h_deltas.shape[0], h_deltas.shape[1]))

            # Hidden to output weight update
            u_o = self.LR * np.dot(np.transpose(self.n_h), o_deltas) + self.ALPHA * u_o
            self.w_o += u_o
            # Input to hidden weight update
            u_h = self.LR * np.dot(np.transpose(self.x), h_deltas[:,:-1]) + self.ALPHA * u_h
            self.w_h += u_h
#            # h_update = LR * (np.dot(inputs),h_deltas) + ALPHA * h_update
            confusion = np.dot(np.transpose(self.t), self.n_o)
            # print(self.t)
            # print(self.CORRECT)
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

    # print('inputs: %s x %s' % (inputs.shape[0], inputs.shape[1]))
    # print('targets: %s x %s' % (targets.shape[0], targets.shape[1]))
    # print('oneurons: %s x %s' % (oneurons.shape[0], oneurons.shape[1]))
    # print('hneurons: %s x %s' % (hneurons.shape[0], hneurons.shape[1]))
    # print('oweights: %s x %s' % (oweights.shape[0], oweights.shape[1]))
    # print('hweights: %s x %s' % (hweights.shape[0], hweights.shape[1]))
    # print('oupdates: %s x %s' % (oupdates.shape[0], oupdates.shape[1]))
    # print('hupdates: %s x %s' % (hupdates.shape[0], hupdates.shape[1]))

if __name__ == '__main__':
    main()
