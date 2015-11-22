import numpy as np, matplotlib.pyplot as mplt, pylab

def createDataAsXYL(numberperset=60):
   data = [[0, 0, 0]]
   covariancematrix = [[.1,0,0],[0,.1,0],[0,0,0]]
   for i in range(numberperset):
      data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 1, 1], covariancematrix) + np.random.multivariate_normal([ 1, 0, 1], covariancematrix))], axis=0)
      data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 0,-1], covariancematrix) + np.random.multivariate_normal([ 1, 1,-1], covariancematrix))], axis=0)
      
   return data[1:].T

def getClassifiedAsFromData(data, label):
   return data.T[data[2]==label].T

data = createDataAsXYL(120)

#6.1a

#6.1b
mplt.scatter(getClassifiedAsFromData(data,  1)[0], getClassifiedAsFromData(data,  1)[1], color='blue')
mplt.scatter(getClassifiedAsFromData(data, -1)[0], getClassifiedAsFromData(data, -1)[1], color='green')
mplt.savefig('6.1b.png', bbox_inches='tight')