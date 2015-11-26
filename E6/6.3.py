from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def y(p, w, sd, reps):
    return np.sign(np.dot(w.T,Phi(p, sd, reps)))

def Phi(p, sd, reps):
    return np.array([Psi(reps[i], sd, p) for i in range(k)]+[1.])

def Psi(basis, sd, p):
    return np.exp((np.linalg.norm(p-basis)**2)/(2*sd**2))

ks = [2,4]
sds = [0.5,0.6]
numberperset=60
my = np.array([[0,1],[1,0],[0,0],[1,1]])
data = [[None, None, None] for x in range(2*numberperset)]
sd = np.sqrt(.1)
toggle = [1,0]

for i in range(0, 2*numberperset, 2):
    myrand = np.random.randint(0,2)
    data[i] = [np.random.normal(myrand, sd), np.random.normal(toggle[myrand], sd)]
    myrand = np.random.randint(0,2)
    data[i+1] = [np.random.normal(myrand, sd), np.random.normal(myrand, sd)]

data = np.array(data)
target = np.array([1.,-1.]*numberperset)

for k in ks:
    cluster = KMeans(k)
    cluster.fit(data)
    reps = cluster.cluster_centers_
    for sd in sds:
        design_matrix = np.array([np.exp((np.linalg.norm(data-my[i],axis=1)**2)/(2*sd**2)) for i in range(k)]+[np.array([1. for i in range(numberperset*2)])])
        weights = np.dot(np.linalg.pinv(design_matrix).T,target)
        print sd, k
        for i in range(10):
            print y(data[i],weights, sd, reps)
