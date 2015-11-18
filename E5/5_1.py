import numpy as np
import numpy.polynomial.polynomial as npp
import pylab as pl
import matplotlib.pyplot as plt
p_1 = []
p_5 = []
p_9 = []
p_14 = []
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
p_1_bias = np.mean((y_hat_1_avg - f(x))**2)
p_1_var = np.mean(np.mean((y_hat_1 - y_hat_1_avg)**2))
p_5_bias = np.mean((y_hat_5_avg - f(x))**2)
p_5_var = np.mean(np.mean((y_hat_5 - y_hat_5_avg)**2))
p_9_bias = np.mean((y_hat_9_avg - f(x))**2)
p_9_var = np.mean(np.mean((y_hat_9 - y_hat_9_avg)**2))
p_14_bias = np.mean((y_hat_14_avg - f(x))**2)
p_14_var = np.mean(np.mean((y_hat_14 - y_hat_14_avg)**2))

#c
plt.figure(1)
plt.plot(x, y_hat_1.T)
plt.plot(x, y_hat_1_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values', linewidth=3)
plt.legend()
plt.title('Prediction, deg(p(x))=1')
pl.savefig('Prediction_d=1.png', bbox_inches='tight')

plt.figure(2)
plt.plot(x, y_hat_5.T)
plt.plot(x, y_hat_5_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values', linewidth=3)
plt.legend()
plt.title('Prediction, deg(p(x))=5')
pl.savefig('Prediction_d=5.png', bbox_inches='tight')

plt.figure(3)
plt.plot(x, y_hat_9.T)
plt.plot(x, y_hat_9_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values', linewidth=3)
plt.legend()
plt.title('Prediction, deg(p(x))=9')
pl.savefig('Prediction_d=9.png', bbox_inches='tight')

plt.figure(4)
plt.plot(x, y_hat_14.T)
plt.plot(x, y_hat_14_avg , label='Average', c='black',linewidth=3)
plt.plot(x, f(x), c='g', label='True values', linewidth=3)
plt.legend()
plt.title('Prediction, deg(p(x))=14')
pl.savefig('Prediction_d=14.png', bbox_inches='tight')

fig, ax = plt.subplots()
width = 0.35
bias = ax.bar(np.arange(4), (p_1_bias, p_5_bias, p_9_bias, p_14_bias), width, color='r')
variances = ax.bar(np.arange(4)+width, (p_1_var, p_5_var, p_9_var, p_14_var), width, color='b')
bias_and_variances = ax.bar(np.arange(4)+2*width, (p_1_bias+p_1_var, p_5_bias+p_5_var, p_9_bias+p_9_var, p_14_bias+p_14_var), width, color='g')
plt.title('Bias and variance for the models')
ax.set_xticks(np.arange(4)+2*width)
ax.set_xticklabels(('deg(p)=1', 'deg(p)=5', 'deg(p)=9', 'deg(p)=14'))
ax.legend((bias, variances, bias_and_variances), ('Biases', 'Variances', 'Biases+Variances'))
pl.savefig('Bias_var.png')
