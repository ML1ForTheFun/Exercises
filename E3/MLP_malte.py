import numpy as np
import matplotlib.pyplot as mplt
import pylab

def NNL(hiddenneurons, a_f, a_fderiv, data, run_number):
    #Training data
    (x, t) = data
    p = len(x)
    nhu = hiddenneurons
    x = x.reshape(p,1)
    t = t.reshape(p,1)

    #Weights and biases are set randomly in the range [-0.5, 0.5);
    w = np.asarray([0.5 * np.random.randn(nhu), 0.5 * np.random.randn(nhu)]);
    b = np.asarray([0.5 * np.random.randn(nhu), 0.5 * np.random.randn(1)]);
    #b = np.asarray([[1.,1.,1.],[1.]])

    a = a_f(np.dot(x,w[0].reshape(1,nhu))-b[0]);
    o = (np.sum(a*np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu)),axis=1) - b[1]*np.ones(p)).reshape(p,1);

    MeanErrors = [0]

    for i in range(12000):
       #Error cost function
       e = 0.5*(o-t)**2;

       MeanErrors.append(np.mean(e))

       #break criterium
       if np.abs(MeanErrors[-1]-MeanErrors[-2])/MeanErrors[-1]<10**-5:
           break

       #Delta_j^v for calculating the weight shift; delta for output is 1 for the identity transfer function
       #d  = 4*np.cosh(np.dot(x,w[0].reshape(1,nhu)))**2/np.cosh(2*np.dot(x,w[0].reshape(1,nhu))+1)**2 * np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu))
       d  = (1-np.tanh(np.dot(x,w[0].reshape(1,nhu))))**2 * np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu))
       #          bias->hiddenunit                    hiddenunit->output
       #db = 4*np.cosh(b[0])**2/np.cosh(2*b[0]+1)**2 * w[1]
       db = (1-np.tanh(b[0]))**2
       #d = a_fderiv(np.dot(x, w[0].reshape(1, nhu)))*np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu))
       #weight adjustion
       dw1 = np.mean((o-t)*a,axis=0)
       dw0 = np.mean((o-t)*d*x,axis=0)
       dwb = np.mean((o-t)*db,axis=0)

       w[0]-=0.05*dw0
       w[1]-=0.05*dw1
       b[0]-=0.05*dwb

       #print("----\n"+str(np.dot(x,w[0].reshape(1,nhu))))
       a = np.tanh(np.dot(x,w[0].reshape(1,nhu))-b[0]);
       #print("----\n"+str(a)+"\n----\n"+str(b[0]))
       o = (np.sum(a*np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu)),axis=1) - b[1]*np.ones(p)).reshape(p,1);

    #a
    iterations = range(len(MeanErrors))
    mplt.plot(iterations,MeanErrors)
    pylab.savefig('./plots/3_1a_t_'+str(run_number)+'.png', bbox_inches='tight')
    mplt.clf()

    
    #b
    x = np.linspace(0.,1.,100).reshape(100,1)
    a = np.tanh(np.dot(x,w[0].reshape(1,nhu))-b[0]);
    for i in range(3):
       mplt.plot(x,a[:,i])
    pylab.savefig('./plots/3_1b_t_'+str(run_number)+'.png', bbox_inches='tight')
    mplt.clf()


    #c
    o = (np.sum(a*np.dot(np.ones(100).reshape(100,1),w[1].reshape(1,nhu)),axis=1) - b[1]*np.ones(100)).reshape(100,1);
    mplt.plot(x,o)
    (x, t) = data
    x = x.reshape(p,1)
    t = t.reshape(p,1)
    mplt.scatter(x,t)
    pylab.savefig('./plots/3_1c_t_'+str(run_number)+'.png', bbox_inches='tight')
    mplt.clf()

    return o[-1]

NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 0)

NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 1)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 2)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 3)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 4)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 5)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 6)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 7)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 8)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 9)
NNL(3, np.tanh, "a_fderiv", np.genfromtxt('RegressionData.txt').T, 10)
#'''





