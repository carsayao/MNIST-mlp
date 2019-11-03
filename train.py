import os
import sys
import warnings
from mlxtend.data import loadlocal_mnist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


class mlp:
    # title, inputs, samples, hidden units, learning rate, momentum, decay, epochs
    def __init__(self, title, inputs, N, hidden, lr, mom, decay, epochs):
        self.TITLE = title
        self.INPUTS = inputs
        self.NEURONS = 10
        self.HIDDEN = hidden
        self.SAMPLES = N
        self.TEST_SAMPLES = 10000
        self.EPOCHS = epochs
        self.LR = lr
        self.ALPHA = mom
        self.LAMBDA = decay
        # Array of epochs to store correct %
        self.CORRECT = []
        self.CORRECT_CONF = np.zeros((10,10))
        # Separate array to store testing accuracy
        self.CORRECT_TEST = []
        self.CORRECT_TEST_CONF = np.zeros((10,10))
        # Absolute path
        self.path = os.path.dirname(os.path.realpath(__file__))
        # Relative paths
        self.train_images_dat = self.path + "/MNIST/train_images.dat"
        self.train_labels_dat = self.path + "/MNIST/train_labels.dat"
        self.test_images_dat = self.path + "/MNIST/test_images.dat"
        self.test_labels_dat = self.path + "/MNIST/test_labels.dat"

        # Set up randomized weights from -.05 to .05
        self.w_o = np.random.uniform(-0.05,0.05,
                                     size=(self.HIDDEN+1,self.NEURONS))
        self.w_h = np.random.uniform(-0.05,0.05,
                                     size=(self.INPUTS+1,self.HIDDEN))

    # Save our plots and training confirmation matrix
    def save(self):
        save_stats = ("hu" + f"{int(self.HIDDEN):03d}"
                    + "-m" + f"{int(self.ALPHA*1000):03d}"
                    + "-ex" + str(self.SAMPLES))

        ax = plt.subplot(111)
        xaxis = np.arange(1, self.EPOCHS+1, 1)
        for n in [self.CORRECT, self.CORRECT_TEST]:
            plt.plot(xaxis, n)
        title_info = (self.TITLE
                   + "; n=" + str(self.HIDDEN)
                   + ", alpha=" + str(self.ALPHA))
        ax.set(xlabel="epoch", ylabel="accuracy", title=title_info)
        ax.legend(("train", "test"), loc="lower right")
        plt.xlim(xmax=self.EPOCHS+1, xmin=0)
        # plt.ylim(ymax=1, ymin=0)
        # ax.yaxis.set_ticks(np.arange(0, 1, .1))
        ax.grid()

        plt.savefig(self.path + "/MNIST/" + save_stats + ".png")

        # Save confusion matrices
        save_conf = (self.path + "/MNIST/train_conf_" + save_stats + ".csv")
        np.savetxt(fname=save_conf, X=self.CORRECT_CONF, fmt='% 5d')
        save_conf = (self.path + "/MNIST/test_conf_" + save_stats + ".csv")
        np.savetxt(fname=save_conf, X=self.CORRECT_TEST_CONF, fmt='% 5d')

    # Load training images and labels from data and add bias to images
    def load(self):
        load_inputs  = np.memmap(self.train_images_dat,
                                 dtype="float64",
                                 mode='r',
                                 shape=(60000,784))
        load_targets = np.memmap(self.train_labels_dat,
                                 dtype="float64",
                                 mode='r',
                                 shape=(60000,10))
        copy_inputs, copy_targets = self.shuffle_sets(load_inputs,
                                                      load_targets)
        return copy_inputs[:self.SAMPLES,:], copy_targets[:self.SAMPLES,:]
    
    def load_test(self):
        load_inputs  = np.memmap(self.train_images_dat,
                                 dtype="float64", mode='r',
                                 shape=(60000,784))
        load_targets = np.memmap(self.train_labels_dat,
                                 dtype="float64",
                                 mode='r',
                                 shape=(60000,10))
        return load_inputs[:self.SAMPLES,:], load_targets[:self.SAMPLES,:]
    
    # Sigmoid function
    def activate(self, n):
        return expit(n)

    def shuffle_sets(self, samples, targets):
        rng_state = np.random.get_state()
        copy_samples = np.empty((np.shape(samples)))
        copy_samples[:] = samples
        np.random.shuffle(copy_samples)
        np.random.set_state(rng_state)
        copy_targets = np.empty((np.shape(targets)))
        copy_targets[:] = targets
        np.random.shuffle(copy_targets)
        return copy_samples[:self.SAMPLES,:], copy_targets[:self.SAMPLES,:]
    
    def test(self):
        test_images, test_labels = self.load_test()
        # Set test sample size
        self.TEST_SAMPLES = test_labels.shape[0]
        # Bias
        test_inputs = np.c_[test_images, np.ones(self.TEST_SAMPLES)]
        # Copies
        test_target_array = np.zeros((np.shape(test_labels)))
        np.copyto(test_target_array, test_labels)
        # Init predictions array
        test_predictions = np.zeros((self.TEST_SAMPLES,self.NEURONS))
        # Test samples but this time only with forward prop
        for N in range(self.TEST_SAMPLES):
            test_o, test_tar_k, test_prediction_n = self.forward(test_inputs[N],
                                                                 test_target_array[N])
            test_predictions[N] = test_prediction_n
        self.CORRECT_TEST_CONF, acc = self.get_confusion(test_target_array,
                                                         test_predictions)
        self.CORRECT_TEST.append(acc)
        print(self.CORRECT_TEST)

    # Params: x==inputs, t==targets, n==neurons, w==weights, u==update,
    #         h==hidden, o==output
    def train(self):
        # Initialize prev update arrays
        u_h = np.zeros((np.shape(self.w_h)))
        u_o = np.zeros((np.shape(self.w_o)))

        for e in range(self.EPOCHS):
            x, t = self.load()
            # Add bias, copy arrays over
            inputs = np.c_[x, np.ones(self.SAMPLES)]
            target_array = np.zeros((np.shape(t)))
            np.copyto(target_array, t)

            # Hold onto our predictions
            predictions = np.zeros((self.SAMPLES,self.NEURONS))

            print("\n\nEpoch:%s lr:%s n:%s N:%s decay:%s"
                   % (e+1, self.LR, self.HIDDEN, self.SAMPLES, self.LAMBDA))

            for N in range(self.SAMPLES):

                # Forward prop
                n_o, target_k, prediction_n = self.forward(inputs[N], target_array[N])

                # Backprop
                # Summations on weights use dots of hidden output errors

                # Output deltas
                o_deltas = n_o * (1-n_o) * (target_k-n_o)
                # Hidden deltas
                h_deltas = self.n_h * (1-self.n_h) * np.dot(o_deltas, np.transpose(self.w_o))

                # Hidden to output weight update
                u_o = (self.LR
                    * (np.dot((np.transpose(np.array(self.n_h)[np.newaxis])),
                           np.array(o_deltas)[np.newaxis]))
                    + (self.ALPHA * u_o))
                # Update
                self.w_o += u_o

                # Input to hidden weight update
                u_h = (self.LR
                    * (np.dot((np.transpose((inputs[N])[np.newaxis])),
                           np.array(h_deltas[:-1])[np.newaxis]))
                    + (self.ALPHA * u_h))
                self.w_h += u_h

                # Add predictions of sample to predictions array to compute confusion matrix
                predictions[N] = prediction_n

            # Append the returned accuracy to correctness array
            self.CORRECT_CONF, train_acc = self.get_confusion(target_array, predictions)
            self.CORRECT.append(train_acc)
            print(self.CORRECT)

            # Run tests after every epoch
            print("\nRunning test\n")
            self.test()

        # Save results after finished epochs
        self.save()

    def forward(self, inputs, target_array):
        # Get hidden units
        self.n_h = np.dot(inputs, self.w_h)
        # Sigmoid
        self.n_h = self.activate(self.n_h)
        # Add bias
        self.n_h = np.append(self.n_h, 1)
        # Get outputs
        n_o = np.dot(self.n_h, self.w_o)

        # 1 Activate max of outputs by storing activations of n_o in predictions array.
        #     * predictions array is used to compute confusion matrix.
        # Init predictions array to shape of output neurons
        predictions = np.empty((np.shape(n_o)))
        predictions = self.activate(n_o)
        # 2 Activate on output neurons.
        n_o = self.activate(n_o)
        # 3 Find the max of each sample in predictions and give it a 1 there, 0 otherwise.
        predictions = np.where(predictions>=np.amax(predictions),1,0)
        target_k = predictions * target_array
        # 4 Turn target array into array of 0.9s and 0.1s.
        #     * for calculating deltas.
        target_k = np.where(target_k==1,0.9,0.1)
        return n_o, target_k, predictions

    # Write to confusion matrix
    def get_confusion(self, target, pred):
        conf = np.array(np.dot(np.transpose(target), pred))
        accuracy = np.trace(conf) / self.SAMPLES
        print("  %:", round(accuracy*100, 4))
        return conf, accuracy

def main():
    # title, inputs, samples, hidden units, learning rate, momentum, decay, epochs
    title = "Experiment 1: Vary Hidden Units"
    hiddens = 20
    momentu = .9
    epochs  = 50
    inputs  = 784
    samples = 60000

    learnin = .1
    decay   = .5
    print("epochs:%s samples:%s hiddens:%s lr:%s alpha:%s"
          %s
         (epochs, samples, hiddens, learnin, momentu)
    run = mlp(title, inputs, samples, hiddens, learnin, momentu, decay, epochs)
    run.train()

if __name__ == "__main__":
    main()
