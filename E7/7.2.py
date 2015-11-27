import numpy as np, matplotlib.pyplot as mplt, pylab


# ------- Data Setup -------
def createDataAsXYL(numberperset=60):
   data = [[None, None, None] for x in range(2*numberperset)]
   covariance_matrix = [[2,0],[0,2]]
   for i in range(0, 2*numberperset, 2):
      data[i] = np.append(np.random.multivariate_normal([0, 1], covariance_matrix), -1)
      data[i+1] = np.append(np.random.multivariate_normal([1, 0], covariance_matrix), 1)
      
   return np.array(data)

def getDataWithLabel(data, label):
   return data[data.T[2]==label]

def computeMinimumWeights(data):
   #(XTX)^-1 XTy
   x = np.asmatrix(data.T[0:2]).T   #120x2
   y = np.asmatrix(data.T[2]).T     #120x1
   
   #print (x.T.dot(x)).shape                                #2x120 . 120x2 = 2x2
   #print np.linalg.inv(x.T.dot(x)).dot(x.T).shape          #2x2 . 2x120 = 2x120
   #print np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y).shape   #2x120 . 120x1 = 2x1
   
   return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


data = createDataAsXYL(60)

#print data
print computeMinimumWeights(data)
