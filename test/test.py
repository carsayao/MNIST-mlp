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
# arg = "tuple"

path = os.path.dirname(os.path.realpath(__file__))
test_images_dat = path + '/../MNIST/test_images.dat'

if arg == 'array':
# if True:
    # newfp = np.memmap(test_images_dat, dtype='float32', shape=(SAMPLES_T,INPUTS))
    # newfp[:,1:] = newfp[:,1:]
    matrix = np.array([[1,2,3,4]
                      ,[5,6,7,8]])
    # print(matrix) 
    # print(matrix.shape[0], matrix.shape[1])
    # sum = np.sum(matrix,axis=1)
    # print('np.sum(matrix),axis=1 : %s' % (sum))
    # matrix = np.c_[matrix, np.ones(2)]
    # print(matrix)
    # print(matrix.shape[0], matrix.shape[1])

    # print('exp\n',np.exp(matrix))
    # print('np.sum(np.exp(matrix),axis=1)\n',np.sum(np.exp(matrix),axis=1))
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
    

if arg == 'format':
    print('hello ' + f'{int(ALPHA*1000):03d}')

if arg == "shuffle":
    samples = np.array([[.01,.05,.09,.04],
                        [.02,.06,.07,.08],
                        [.03,.01,.11,.12],
                        [.04,1,1,1],
                        [.05,1,2,3]])
    targets = np.array([[1, 5],
                        [2, 6],
                        [3, 1],
                        [4,1],
                        [5,1]])
    print("samples")
    print(samples)
    print("targets")
    print(targets)
    print(np.shape(samples)[0])
    # ran = range(np.shape(samples)[0])
    # print(ran)
    # np.random.shuffle(ran)
    print(samples)
    def shuffle_sets(samples, targets):
        rng_state = np.random.get_state()
        # print("rng_state",rng_state)
        np.random.shuffle(samples)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)
    shuffle_sets(samples, targets)
    print(" ----- SHUFFLE ----- ")
    print("samples")
    print(samples)
    print("targets")
    print(targets)
    print("shape [0,1,2,3]")
    arr = np.array([0,1,2,3])
    print(arr)
    # np.reshape(arr,(-1,1))
    # arr = np.transpose(np.matrix(arr))
    # arr = np.transpose(np.array(arr))
    print("reshape")
    arr = np.array(arr)[np.newaxis]
    arr = np.transpose(arr)
    print(arr)

    # pended = np.append(arr, 1)
    # print(pended)
    # print(pended[:-1])
    # pred = np.empty((3,2))
    pred = []
    ones = [0,0]
    # print(pred)

    # appred = np.append(pred,ones, axis=1)
    # appred = np.append([appred],[ones], axis=1)
    # appred = np.append([appred],[ones], axis=1)
    # appred = np.append([appred],[ones], axis=1)
    # appred = np.append([appred],[ones], axis=1)
    # appred = np.append([appred],[ones], axis=1)
    # print(appred)

if arg == "tuple":
    s = 4
    t = 4
    x = 3
    n = 2
    inputs = np.ones((s,x))
    print("inputssss\n",inputs)
    targets = np.zeros((s,n))
    print("targets\n",targets)
    print(np.asarray(inputs))
    shuffle = range(s)
    print(shuffle)
    # np.random.shuffle(shuffle)
    print(shuffle)
    # tup = np.array((np.asarray(inputs),np.asarray(targets)))
    # print("tup\n",tup)

if arg=="dumb":
    matrix = np.ones((3,5))
    a = np.ones(np.shape(matrix))
    b = np.ones((np.shape(matrix)))
    print("matrix\n",matrix)
    print("a\n",a)
    print("b\b",b)

if arg=="pred":
    s = 4
    n_o = np.array([[0,0,1,0],
                    [0,0,0,1],
                    [1,0,0,0],
                    [0,1,0,0]])
    t   = np.array([[0,0,1,0],
                    [0,0,1,0],
                    [0,1,0,0],
                    [1,0,0,0]], dtype="float32")
    print("n_o")
    print(n_o)
    print("t")
    print(t)
    print("n_o*t")
    target_k = n_o*t
    print(n_o*t)
    for N in range(s):
        # target_k[N] = np.where(target_k[N]>=np.amax(target_k[N]),0.9,0.1)
        target_k[N] = np.where(target_k[N]==1,.9,.1)
    print("target_k")
    print(target_k)
    matrix = np.array([[1,-2,3,4],
                       [5,-6,7,8],
                       [9,-10,11,12]])
    print(np.transpose(matrix))
    # print(-matrix)

if arg=="slice":
    # h_deltas = np.ones((SAMPLES,NEURONS_H+1))
    h_deltas = np.array([[1,2,3,4],
                         [2,6,7,8],
                         [3,1,2,3],
                         [4,8,7,6],
                         [5,1,1,1]])
    zer = np.zeros(h_deltas.shape[0])

    print('\nh_deltas:%sx%s' % (h_deltas.shape[0],h_deltas.shape[1]))
    print(h_deltas)
    print("\nzer")
    print(zer, '\n')

    h_deltas = np.c_[h_deltas,zer]
    print('\nh_deltas dim after np.c_[]:%sx%s' % (h_deltas.shape[0],h_deltas.shape[1]))
    print(h_deltas)

    # Remove bottom row
    chop = h_deltas[:-1:]
    print('\nchop:%sx%s' % (chop.shape[0],chop.shape[1]))
    print(chop, '\n')

    # Remove bottom row
    chop = h_deltas[:-1,:]
    print('\nchop:%sx%s' % (chop.shape[0],chop.shape[1]))
    print(chop, '\n')

    # Remove first n col
    chop = h_deltas[:3,:]
    print('\nchop:%sx%s' % (chop.shape[0],chop.shape[1]))
    print(chop, '\n')

    # # Remove first 2 col
    # chop = h_deltas[:,2:]
    # print('\nchop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')

    # # Preserve 1st row
    # chop = h_deltas[:1,:]
    # print('\nchop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')
    
    # # Remove 1st and last col, preserve first row
    # chop = h_deltas[:,1:-1]
    # print('\nchop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop[0], '\n')

    # # Chop first column
    # chop = h_deltas[:,1:]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')

    # # Chop last column
    # chop = h_deltas[:,:-1]
    # print('chop:%sx%s' % (chop.shape[0],chop.shape[1]))
    # print(chop, '\n')

if arg=="copy":
    orig = np.ones((3,5))
    col  = np.zeros(3)
    copy = np.c_[orig,col]
    print(copy)
    print(orig)
    orig = np.c_[copy,col]
    print(copy)
    print(orig)

    def reassign(alist):
        alist = [0,1]
    alist = [0]
    reassign(alist)
    print("\nalist",alist)

    def reassignret(blist):
        blist = [0,1]
        return blist
    blist = [0]
    blist = reassignret(blist)
    print("\nblist",blist)

    lista = [0]
    listb = lista
    listb.append(1)
    print("\nlista",lista)