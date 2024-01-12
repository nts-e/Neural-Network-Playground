# Neural-Network-Playground

Neural network development from scratch. A simple 4-layer network with forward/backward propagation, dropout, and batch normalization.

The code implements the following network:

<img width="550" alt="image" src="https://github.com/nts-e/Neural-Network-Playground/assets/107881111/c61018d1-9caa-440e-a11c-d66af98a0055">


The network weights have been initialized as recommended: l and l-1 are the number of neurons in the current layer and in the previous layer respectively:
np.random.randn(l, layer_dims[i-1]) * np.sqrt(2/l)
This initialization generates normally distributed random numbers with mean 0 and variance 1. Multiplying this array by np.sqrt(2/l) scales the variance of the random numbers by a factor of 2/l, to ensure that the weight initialization is neither
too small, which can lead to vanishing gradients and slow convergence, nor too large, which can cause exploding gradients and unstable learning. I tried initializing the weights using other techniques, such as random initialization multiplied by 0.01 or by 1. However, the network did not learn in these cases, highlighting the importance of proper weight initialization for effective learning.



In addition, the program incorporates a dropout mechanism for hiding node activations in hidden layers. Initialization involves passing a list of dropout probabilities, "layers_dropout," to the L_layer_model function. Each relu function receives a dropout value for layer use. The relu function carries out dropout in three steps: generating a mask vector with random 1s and 0s, multiplying it by a calculated factor, and applying this factored mask to the minibatch. During backpropagation, the relu_backward function zeroes out gradients corresponding to dropped out activations, ensuring that only active activations contribute to the gradient computation, aligning with dropout regularization principles.







