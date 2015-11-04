#Initialization
import numpy as np

#Training data
(x, t) = np.genfromtxt('RegressionData.txt').T;
x = x.reshape(10,1)
t = t.reshape(10,1)

#Weights and biases are set randomly in the range [-0.5, 0.5);
w = np.asarray([0.5 * np.random.randn(3), 0.5 * np.random.randn(3)]);
b = np.asarray([0.5 * np.random.randn(3), 0.5 * np.random.randn(1)]);
#b = np.asarray([[1.,1.,1.],[1.]])

a = np.tanh(np.dot(x,w[0].reshape(1,3))-b[0]);
o = (np.mean(a*np.dot(np.ones(10).reshape(10,1),w[1].reshape(1,3)),axis=1) - b[1]*np.ones(10)).reshape(10,1);

for i in range(3000):
        
    #Error cost function
    e = 0.5*(o-t)**2;
   
    #forward propagation
    a = np.tanh(np.dot(x,w[0].reshape(1,3))-b[0]);
    o = (np.mean(a*np.dot(np.ones(10).reshape(10,1),w[1].reshape(1,3)),axis=1) - b[1]*np.ones(10)).reshape(10,1);
	
	#break criterium
    if np.abs(np.max((e-0.5*(o-t)**2)/e))<10**-5:  
        break
    
    #Delta_j^v for calculating the weight shift; delta for output is 1 for the identity transfer function
    d = 4*np.cosh(np.dot(x,w[0].reshape(1,3)))**2/np.cosh(2*np.dot(x,w[0].reshape(1,3))+1)**2*np.dot(np.ones(10).reshape(10,1),w[1].reshape(1,3))
    
    #weight adjustion
    dw1 = np.mean((o-t)*a,axis=0)
    dw0 = np.mean((o-t)*d*x,axis=0)
    
    w[0]-=0.05*dw0
    w[1]-=0.05*dw1

print o-t
