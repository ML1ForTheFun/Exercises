import numpy as np, matplotlib.pyplot as mplt, pylab
from scipy.spatial import cKDTree as tree
from random import randint

# ------- Data Setup -------
def createDataAsXYL(numberperset=60):
   data = [[None, None, None] for x in range(2*numberperset)]
   sd = np.sqrt(.1)
   for i in range(0, 2*numberperset, 2):
      myrand = randint(0,1)
      data[i] = [np.random.normal(myrand, sd), np.random.normal(not myrand, sd),  1]
      myrand = randint(0,1)
      data[i+1] = [np.random.normal(myrand, sd), np.random.normal(myrand, sd), -1]
      
   return np.array(data)

def getDataWithLabel(data, label):
   return data[data.T[2]==label]

data = createDataAsXYL(60)
#print data

# ------- 6.1 -------
def findNearestNeighborIndiciesToScipy(data, pointindex, numberofneighbors):
   t = tree(data[:,:-1])
   return t.query(data[:,:-1][pointindex], numberofneighbors+1)[1][1:]


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



# ------- 6.2 -------
def findClassificationFromDataAndCenterindex(data, centerpointindex, sigmasquared):
   cumulativevote = 0
   #print str(centerpointindex)+" --- "+str(data[centerpointindex])
   for otherpointindex in range(len(data)):
      #print str(centerpointindex)+" --- "+str(otherpointindex)+" --- "+str(data[otherpointindex])
      if otherpointindex==centerpointindex:
         continue
      
      #find weight for this point to other point
      weight = computeWeightFromDistance(\
                                          computeDistance(data[centerpointindex][0], data[centerpointindex][1], data[otherpointindex][0], data[otherpointindex][1]),\
                                          sigmasquared\
                                       )
      #print "\t", weight, data[centerpointindex][2]
      cumulativevote += weight * data[otherpointindex][2]
   
   return cumulativevote
   
def computeDistance(ax, ay, bx, by):
   return np.sqrt((ax-bx)**2 + (ay-by)**2)
   
def computeWeightFromDistance(distance, sigmasquared):
   return np.exp(-1 * (distance**2 / (2*sigmasquared)) )
   

for sigmasquared in [.5, .1, .01]:
   classifiedasones = [None] * len(data)
   
   #find the classifications
   for centerpointindex in range(len(data)):
      cumulativevote = findClassificationFromDataAndCenterindex(data, centerpointindex, sigmasquared)
      classifiedasones[centerpointindex] = cumulativevote > 0
   
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
   mplt.savefig('6.2b_s2='+str(sigmasquared)+'.png', bbox_inches='tight')