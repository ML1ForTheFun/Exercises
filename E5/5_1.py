import numpy as np
import numpy.polynomial.polynomial as npp
import matplotlib.pyplot as plt
p_1 = []
p_5 = []
p_9 = []
p_14 = []
y = []
N = 51
x = np.linspace(0, 1, N)
def f(t):
    return -np.sin(5*np.pi*t)+2*(2*t-1)**3
#a)
y = f(x) + np.random.normal(0, 1, (40, len(x)))
#polynomial fittings
p_1 = npp.polyfit(x, y.T, 1)
p_5 = npp.polyfit(x, y.T, 5)
p_9 = npp.polyfit(x, y.T, 9)
p_14 = npp.polyfit(x, y.T, 14)
#b)
#predicted values with mean and covariance along the 40 samples
y_hat = npp.polyval(x, p_1) 
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
