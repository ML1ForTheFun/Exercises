import numpy as np, matplotlib.pyplot as mplt, pylab
from scipy.spatial import cKDTree as tree
from random import randint

def createDataAsXYL(numberperset=60):
   #data = []
   #covariancematrix = [[.2,0,0],[0,.2,0],[0,0,0]]
   data = [[None, None, None] for x in range(2*numberperset)]
   sd = np.sqrt(.1)
   for i in range(0, 2*numberperset, 2):
      #expected this to work
      #data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 1, 1], covariancematrix) + np.random.multivariate_normal([ 1, 0, 1], covariancematrix))], axis=0)
      #data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 0,-1], covariancematrix) + np.random.multivariate_normal([ 1, 1,-1], covariancematrix))], axis=0)
      
      #hacky solution
      myrand = randint(0,1)
      data[i] = [np.random.normal(myrand, sd), np.random.normal(not myrand, sd),  1]
      myrand = randint(0,1)
      data[i+1] = [np.random.normal(myrand, sd), np.random.normal(myrand, sd), -1]
      
   return np.array(data)

def getDataWithLabel(data, label):
   return data[data.T[2]==label]

def findNearestNeighborIndiciesToScipy(data, pointindex, numberofneighbors):
   t = tree(data[:,:-1])
   return t.query(data[:,:-1][pointindex], numberofneighbors+1)[1][1:]
      
#6.1
data = createDataAsXYL(60)

for k in [1, 3, 5]:
   classifiedasones = [None] * len(data)
   
   #find the classifications
   for centerpointindex in range(len(data)):
      classifications = data[ findNearestNeighborIndiciesToScipy(data, centerpointindex, k) ].T[2]
      onespercentage = np.sum( (classifications + 1)*.5 ) / k
      #print onespercentage > .5, onespercentage, classifications

      classifiedasones[centerpointindex] = onespercentage > .5
   #print np.asarray(classifiedasones)
   #print len(data[np.asarray(classifiedasones)]), len(data[np.asarray(classifiedasones)==False])
   ones = data[np.asarray(classifiedasones)]
   zeros = data[np.asarray(classifiedasones)==False]
   #print "ones:\n"+str(ones)
   #print "zeros:\n"+str(zeros)

   mplt.scatter(getDataWithLabel(zeros,  1).T[0], getDataWithLabel(zeros,  1).T[1], color='red',  marker='s')        #generated as  1, marked as -1
   mplt.scatter(getDataWithLabel(ones,   1).T[0], getDataWithLabel(ones,   1).T[1], color='blue', marker='s')        #generated as  1, marked as  1
   mplt.scatter(getDataWithLabel(zeros, -1).T[0], getDataWithLabel(zeros, -1).T[1], color='blue', marker=r'$\star$') #generated as -1, marked as -1
   mplt.scatter(getDataWithLabel(ones,  -1).T[0], getDataWithLabel(ones,  -1).T[1], color='red',  marker=r'$\star$') #generated as -1, marked as  1
   mplt.savefig('6.1b_k='+str(k)+'.png', bbox_inches='tight')
