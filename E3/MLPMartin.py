#Initialization
import numpy as np
#Training data
(x, t) = np.genfromtxt('RegressionData.txt').T;

#Number of neurons per layer.
n = [1, 3, 1];

#Weights and biases are set randomly in the range [-0.5, 0.5);
w = ([-.5 + np.random.randn(j, i + 1) for i, j in zip(n[:-1], n[1:])]);
#b = list(([np.random.randn(3) - .5], np.random.randn(1) - .5));
#learning step
ls = 0.5
#Add the bias node to the inputs
x = np.array((np.ones(len(x)), x)).T;

#Forward propagation.
#Returns a layer-by-layer list of the activation for neurons
def f_prop(x, w):
    a = a = np.asarray([np.tanh(np.dot(w[0], s)) for s in x]);
    ab = np.ones((len(x), 4));
    ab[:, 1:] = a
    o = np.squeeze(np.asarray([np.dot(w[1], s) for s in ab]));
    return list((a, o));

#The derivative of tanh
def dtanh(x):
    return 4*np.cosh(x)**2/((np.cosh(2*x)+1)**2);

#IF SOMEBODY CAN FIGURE THIS SHIT OUT YO!
#for i in range(3000):
act = f_prop(x, w);
err = ((activations[-1]-t)**2/2);
d = np.vectorize(dtanh)([np.dot(w[0], s) for s in x]);
E = np.mean(e);
for ix in x:
    for j in range(w[0]):
        for i in range(w[0].T)
        dEdw_0[j][i] = (act[1] - t)*ix*d[0][0];
        dEdw_00 = (act[1] - t)*i[0]*d[0][0];

dEdw_2 = (act[1] - t)*act[0]

    #loc_err = backward_prop(x, w, activations[1]);
    #m_loc_err = ([np.mean(err,0) for err in loc_err]);
    #m_act =([np.mean(act,0) for act in activations]);
    #m_err = np.mean(m_act[-1]-t)
    #this is ugly
    for m in range(len(x)):
        for i in range(w[0].shape[0]):
            for j in range(w[0].shape[1]):
                d_w[0][i][j] += (activations[-1][m]-t[m])*(loc_err[0][m][i])*activations[0][m][j];
            for k in range(len(w[1])):
                d_w[1][k] += (activations[-1][m]-t[m])*loc_err[1][m]*activations[1][m][k];
    d_w[0] = d_w[0]/len(x);
    d_w[1] = d_w[1]/len(x);
    print(d_w);
    break

    #update w
    w = ([we - ls*de_w for we, de_w in zip(w, d_w)])
    #Check if criterion is fullfilled
    new_err = np.mean((forward_prop(x, w)[-1]-t)**2/2);
    d_err = abs(new_err-old_err);
    print 'new error', new_err
    print 'old error', old_err
    if (d_err/new_err)<(10**(-5)):
        print 'optimal weights: ', w;
        print 'giving an error rate of: ', new_err
        break
        pause
