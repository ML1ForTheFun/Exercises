import numpy as np
import numpy.polynomial.polynomial as npp
import pylab as pl
import matplotlib.pyplot as plt
p_1 = []
p_5 = []
p_9 = []
p_14 = []
y = []
N = 51
samples = 40
x = np.linspace(0, 1, N)
def f(t):
    return -np.sin(5*np.pi*t)+2*(2*t-1)**3
#a)
y = f(x) + np.random.normal(0, 1, (samples, len(x)))
#polynomial fittings
p_1 = npp.polyfit(x, y.T, 1)
p_5 = npp.polyfit(x, y.T, 5)
p_9 = npp.polyfit(x, y.T, 9)
p_14 = npp.polyfit(x, y.T, 14)
#b)
#predicted values with mean and covariance along the 40 samples
y_hat_1 = npp.polyval(x, p_1) 
y_hat_5 = npp.polyval(x, p_5)
y_hat_9 = npp.polyval(x, p_9)
y_hat_14 = npp.polyval(x, p_14)
y_hat_1_avg = np.mean(y_hat_1, 0)
y_hat_1_var = np.var(y_hat_1, 0)
y_hat_5_avg = np.mean(y_hat_5, 0)
y_hat_5_var = np.var(y_hat_5, 0)
y_hat_9_avg = np.mean(y_hat_9, 0)
y_hat_9_var = np.var(y_hat_9, 0)
y_hat_14_avg = np.mean(y_hat_14, 0)
y_hat_14_var = np.var(y_hat_14, 0)
#Bias and variance of each model
p_1_bias = y_hat_1_avg - f(x)
p_1_var = np.mean((y_hat_1 - y_hat_1_avg)**2, 0)
p_5_bias = y_hat_5_avg - f(x)
p_5_var = np.mean((y_hat_5 - y_hat_5_avg)**2, 0)
p_9_bias = y_hat_9_avg - f(x)
p_9_var = np.mean((y_hat_9 - y_hat_9_avg)**2, 0)
p_14_bias = y_hat_14_avg - f(x)
p_14_var = np.mean((y_hat_14 - y_hat_14_avg)**2, 0)

#c
plt.figure(1)
plt.plot(x, y_hat_1.T)
plt.plot(x, y_hat_1_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values')
plt.legend()
pl.savefig('Prediction_d=1.png', bbox_inches='tight')

plt.figure(2)
plt.plot(x, y_hat_5.T)
plt.plot(x, y_hat_5_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values')
plt.legend()
pl.savefig('Prediction_d=5.png', bbox_inches='tight')

plt.figure(3)
plt.plot(x, y_hat_9.T)
plt.plot(x, y_hat_9_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values')
plt.legend()
pl.savefig('Prediction_d=9.png', bbox_inches='tight')

plt.figure(4)
plt.plot(x, y_hat_14.T)
plt.plot(x, y_hat_14_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values')
plt.legend()
pl.savefig('Prediction_d=14.png', bbox_inches='tight')
