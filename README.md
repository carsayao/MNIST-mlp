# MNIST-mlp
MLP for the MNIST dataset

### log

10/26 ~~Testing sigmoid function.~~
10/27 Working on backwards deltas.
10/28 Weight update uses update from previous backprop.
10/29 Confusion matrix goes deeply negative or positive around 5 epochs. I know the sigmoid function sometimes draws  `RuntimeWarning: overflow encountered in multiply` (near beginning). Need to print some lines of matrices either to .csv's or screen to debug what's happening.

It could also possibly be the way I activate the neurons. I'm still unclear about how we handle target errors on updates. **Investigate further**.

10/30 Convert target and n_o arrays to get max arg, then comput confusion matrix