import MLP_malte
import numpy as np
data = np.genfromtxt('simulatedRFmapping.csv', delimiter=',', skip_header=1);
def dtanh(x):
    return (4*np.cosh(x)**2/((np.cosh(2*x)+1)**2));

def tan1(x):
    return np.vectorize(1+np.tanh(x));

o_1 = MLP_malte.NNL(1,tan1, dtanh, data);
o_2 = MLP_malte.NNL(2,tan1, dtanh, data);
o_3 = MLP_malte.NNL(3,tan1, dtanh, data);
o_4 = MLP_malte.NNL(4,tan1, dtanh, data);
