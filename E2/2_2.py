import matplotlib.pyplot as mplt
import numpy as np
import pylab as pl
mplt.set_cmap('pink_r');
data = np.genfromtxt('applesOranges.csv', delimiter=',', skip_header=1);
x = data[:,:-1].T;
y = data[:,-1];

#a - points are colored acording to their y value
mplt.scatter(x[0], x[1], c=y);
pl.savefig('Input data with classification', bbox_inches='tight');
mplt.clf()
#b
#angles
a = np.linspace(0, np.pi, 19);
w = np.array([np.cos(a), np.sin(a)]);

#output for each pair of weights
o = np.sign(np.dot(w.T, x));
#making the sign from 0 to 1 instead of -1 and 1
o = np.sign(o + 1);
#classification performance of weight pairs
p = np.array([np.mean(out==y) for out in o]);
mplt.plot(a, p);
pl.savefig('Classification performance for various w-angles');
mplt.clf();
#best performing vector
W = w[:,np.argmax(p)]

#c
thetas = np.linspace(-3, 3, 1000).T
#same procedure as before
o = np.sign(np.tile(np.dot(W, x), (len(thetas), 1)).T - thetas);
o = np.sign(o + 1).T;
p = np.array([np.mean(out==y) for out in o]);
#best performing theta
theta = thetas[np.argmax(p)];
pind = max(p);

def f(x, we, thet):
   o = np.sign(np.dot(we, x) - thet);
   return np.sign(o + 1);
#d
mplt.scatter(x[0], x[1], c=f(x, W, theta));
mplt.scatter(W[0], W[1], c='hotpink');
pl.savefig('Classification performance with w and theta optimized separately');
mplt.clf();
#As seen in the figure, the best w is perpendiculair to the line along which x are classified. w*x gives us a measure of how far away from this line our inputs are.
#e
o = np.sign(np.tile(np.dot(w.T, x), (1000, 1, 1)).T - thetas);
o = np.sign(o + 1);
p = np.apply_along_axis(np.mean, 0, (np.apply_along_axis(np.equal, 0, o, y)));
#The best combination of w and theta:
(n, k) = np.unravel_index(np.argmax(p), (19, 1000));
best_p = np.amax(p)
best_w = w[:,n];
best_theta = thetas[k]
print 'Best w: ', best_w, '\nBest theta: ', best_theta, 'gives ', best_p, ' classification rate';
print 'If we optimize w and theta one after another we get a ', pind, ' classification rate';

mplt.scatter(x[0], x[1], c=f(x, W, theta));
pl.savefig('Classification performance with w and theta optimized simoultaneously');
mplt.clf();
