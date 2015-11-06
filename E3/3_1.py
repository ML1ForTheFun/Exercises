#Initilize the first neural network
import MLP_malte
import numpy as np
def dtanh(x):
    return (4*np.cosh(x)**2/((np.cosh(2*x)+1)**2));

data = np.genfromtxt('RegressionData.txt').T;
output = MLP_malte.NNL(3, np.tanh, dtanh, data);
