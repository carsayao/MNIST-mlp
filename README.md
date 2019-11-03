# MNIST-mlp
MLP for the MNIST dataset

## log

### 10/26

~~Testing sigmoid function.~~

### 10/27

Working on backwards deltas.

### 10/28

Weight update uses update from previous backprop.

### 10/29

Confusion matrix goes deeply negative or positive around 5 epochs. I know the sigmoid function sometimes draws  `RuntimeWarning: overflow encountered in multiply` (near beginning). Need to print some lines of matrices either to .csv's or screen to debug what's happening. *Was likely from activation issues.*

It could also possibly be the way I activate the neurons. I'm still unclear about how we handle target errors on updates. **Investigate further**.

### 10/30

Convert `t` (target) and `n_o` arrays to get max arg, then compute confusion matrix. Shuffle the dataset for every epoch. But first, you need to bind every target array to corresponding input vector. 

Was able to get better results (doesn't shoot off to `inf` or `nan`) by converting targets to 0.9 or 0.1 before calculating deltas but % correct stays constant (<20% and stays absolutely constant). Confusion matrix doesn't seem to be a problem. Shuffling could help? 

### 10/31

Added shuffle and weight decay to training function.

On second epoch, output neurons values (dot product of hidden neurons and hidden weights) are too low to go through activation; they just become 0.

This causes `predictions` matrix to become all 1's after the first epoch. This matrix is dotted with the target array to construct a confusion matrix. The confusion matrix gives bad results because of this bug.

Changed `train.py` to read directly from file. Shuffling arrays loaded from `dat` files might've cause some issues.

Need to work on why this confusion matrix is not working. Should start looking at places I modify `predictions`.

Read in all with `memmap`, then take a random sampling of the data.