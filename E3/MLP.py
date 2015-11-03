#Initialization
import numpy as np
#Training data
(x, t) = np.genfromtxt('RegressionData.txt').T;

#Number of neurons per layer.
n = [1, 3, 1];

#Weights and biases are set randomly in the range [-0.5, 0.5);
w = [-.5 + np.random.randn(i, j) for i, j in zip(n[:-1], n[1:])];
b = [np.random.randn(i) - .5 for i in n[1:]];

#Forward propagation
#activation of neurons, layer by layer
a = np.squeeze(np.tanh([np.dot(w[0], i) - b[0] for i in x]));
o = np.squeeze(np.asarray([np.dot(w[1].T, o.T) - b[1] for o in a_1]));

#Error cost function
e = np.asarray([(y_t - y_w)**2/2 for y_t, y_w in zip(t, o)]);
