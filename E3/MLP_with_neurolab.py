import neurolab as nl
import numpy as np

(x, t) = np.genfromtxt('RegressionData.txt').T;

x = x.reshape(10,1)
t = t.reshape(10,1)

net = nl.net.newff([[0.,1.]],[3,1])

# Train network
error = net.train(x, t, epochs=3000, show=1000, goal=0.0002)

# Simulate network
out = net.sim(x)

# Plot result
print out-t
