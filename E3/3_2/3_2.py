import MLP_malte
import numpy as np
data = np.genfromtxt('simulatedRFmapping.csv', delimiter=',', skip_header=1);
def dtanh(x):
    return (4*np.cosh(x)**2/((np.cosh(2*x)+1)**2));

o_1 = MLP_malte.NNL(1,1+np.tanh, dtanh, data);
o_2 = MLP_malte.NNL(2,1+np.tanh, dtanh, data);
o_3 = MLP_malte.NNL(3,1+np.tanh, dtanh, data);
o_4 = MLP_malte.NNL(4,1+np.tanh, dtanh, data);
