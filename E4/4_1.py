import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

w_i = np.random.rand(2).reshape(2,1)
X = np.array(((1, -1), (1, .3), (1,2))).T
T = np.array((-0.1, 0.5, 0.5)).reshape(1,3)
b = -np.dot(X,T.T)
H = np.dot(X,X.T)
n = 0.5

def grad(w):
    return np.dot(H, w) + b

#4.1a
def grad_desc(w, n):
    return w - n*grad(w)
#4.1b
def line_search(w):
    g = grad(w)
    div = np.dot(g.T,np.dot(H, g))
    if div!=0.:
        a = -np.dot(g.T,g)/div
    else:
        a = 0
    return w + a*g
#4.1c
def conj_grad(w, d):
    div = np.dot(d.T,np.dot(H, d))
    if div!=0.:
        a = -np.dot(d.T,grad(w))/div
    else:
        a = 0
    return w + a*d

#4.1 - iterations
i = 10
d = -grad(w_i)
g = [grad(w_i)]
w_a = [w_i]
w_b = [w_i]
w_c = [w_i]
for s in range(1,i):
    w_a.append(grad_desc(w_a[s-1], n))
    w_b.append(line_search(w_b[s-1]))
    w_c.append(conj_grad(w_c[s-1], d))
    g.append(grad(w_c[s]))
    if np.dot(g[s-1].T,g[s-1])!=0.:
        beta = -np.dot(g[s].T,g[s])/np.dot(g[s-1].T,g[s-1])
        d = g[s]+beta*d
    else:
        d = np.array([[0.], [0.]])
    

#4.1d - Visualization
plt.scatter(*zip(*w_a), c='b', label='Gradient descent')
plt.scatter(*zip(*w_b), c='r', label='Line search')
plt.scatter(*zip(*w_c), c='g', label='Conjugate gradient')
plt.plot(*zip(*w_a), c='b')
plt.plot(*zip(*w_b), c='r')
plt.plot(*zip(*w_c), c='g')
pl.xlim([0,1])
pl.ylim([0,1])
plt.legend()
pl.savefig('comparison.png',bbox_inches='tight')
