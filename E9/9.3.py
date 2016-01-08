import numpy as np, matplotlib.pyplot as mplt, pylab
from sklearn.svm import SVC

# ------- 9.3 - Data Setup -------
def createDataAsXYL(numberperset=60):
   data = [[None, None, None] for x in range(2*numberperset)]
   sd = np.sqrt(.1)
   for i in range(0, 2*numberperset, 2):
      myrand = np.random.randint(0,2)
      data[i] = [np.random.normal(myrand, sd), np.random.normal(not myrand, sd),  1]
      myrand = np.random.randint(0,2)
      data[i+1] = [np.random.normal(myrand, sd), np.random.normal(myrand, sd), -1]
      
   return np.array(data)

def getDataWithLabel(data, label):
   return data[data.T[2]==label]

trainingdata = createDataAsXYL(40)
testdata = createDataAsXYL(40)


# ------- 9.3.a train SVM -------
clf = SVC()
clf.fit(trainingdata[:,0:2], trainingdata[:, 2:3].reshape(80,))

# ------- 9.3.b Classify with 0/1 loss function and report classification error -------
errors = (clf.predict(testdata[:,0:2]) != testdata[:, 2:3].reshape(80,))*1
print( errors )
print(sum(errors) / 80)



# ------- 9.3.c visualize as in 6 -------
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


def findClassificationForAllPoints(data, coordinates, sigmasquared):
   cumulativevote = 0
   for dataindex in range(len(data)):
      if np.linalg.norm(coordinates-data[dataindex,:-1])==0:
         continue
      weight = computeWeightFromDistance(np.linalg.norm(coordinates-data[dataindex,:-1]), sigmasquared)
      cumulativevote += weight * data[dataindex][2]
   return cumulativevote



h = .08  # step size in the mesh
# create a mesh to plot in
x_min, x_max = testdata[:, 0].min(), testdata[:, 0].max()
y_min, y_max = testdata[:, 1].min(), testdata[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
allpoints = np.c_[xx.ravel(), yy.ravel()]

sigmasquared = .1
classifiedasones = [None] * len(testdata)

#find the classifications
for centerpointindex in range(len(testdata)):
   cumulativevote = findClassificationFromDataAndCenterindex(testdata, centerpointindex, sigmasquared)
   classifiedasones[centerpointindex] = cumulativevote > 0


ones = testdata[np.asarray(classifiedasones)]
zeros = testdata[np.asarray(classifiedasones)==False]
contours = np.zeros(allpoints.shape[0])
for i in range(len(allpoints)):
   contours[i] = np.sign(findClassificationForAllPoints(testdata, allpoints[i], sigmasquared))


#print "ones:\n"+str(ones)
#print "zeros:\n"+str(zeros)
mplt.figure()
mplt.scatter(getDataWithLabel(zeros,  1).T[0], getDataWithLabel(zeros,  1).T[1], color='red',  marker='s', label='False negatives')        #generated as  1, marked as -1
mplt.scatter(getDataWithLabel(ones,   1).T[0], getDataWithLabel(ones,   1).T[1], color='blue', marker='s', label='True positives')        #generated as  1, marked as  1
mplt.scatter(getDataWithLabel(zeros, -1).T[0], getDataWithLabel(zeros, -1).T[1], color='blue', marker=r'$\star$', label='True negatives') #generated as -1, marked as -1
mplt.scatter(getDataWithLabel(ones,  -1).T[0], getDataWithLabel(ones,  -1).T[1], color='red',  marker=r'$\star$', label='False positives') #generated as -1, marked as  1
mplt.contour(xx, yy, contours.reshape(xx.shape), cmap=mplt.cm.Paired, levels=[0.5]);
mplt.legend(loc=2)
mplt.savefig('9.3.png', bbox_inches='tight')
