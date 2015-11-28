import numpy as np, matplotlib.pyplot as mplt, pylab


# ------- Data Setup -------
def createDataAsXYL(numberperset=60):
   data = [[None, None, None] for x in range(2*numberperset)]
   covariance_matrix = [[2,0],[0,2]]
   for i in range(0, 2*numberperset, 2):
      data[i] = np.append(np.random.multivariate_normal([0, 1], covariance_matrix), -1)
      data[i+1] = np.append(np.random.multivariate_normal([1, 0], covariance_matrix), 1)
      
   return np.array(data)

def getXYDataByClasses():
   return data[data.T[2]==-1][:,0:2], data[data.T[2]== 1][:,0:2]

def getXYTestdataByClassesAndPredictions():
   c1 = testdata[testdata.T[2]==-1]
   c2 = testdata[testdata.T[2]==1]
   return c1[c1.T[3]==-1][:,0:2], c1[c1.T[3]==1][:,0:2], c2[c2.T[3]==-1][:,0:2], c2[c2.T[3]==1][:,0:2]




def computeMinimumWeights(data):
   #(XTX)^-1 XTy
   x = np.asmatrix(data.T[0:2]).T   #120x2
   y = np.asmatrix(data.T[2]).T     #120x1
   
   #print (x.T.dot(x)).shape                                #2x120 . 120x2 = 2x2
   #print np.linalg.inv(x.T.dot(x)).dot(x.T).shape          #2x2 . 2x120 = 2x120
   #print np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y).shape   #2x120 . 120x1 = 2x1
   
   return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

def computeTestDataPredictions(testdata, weights):
   x = np.asmatrix(testdata.T[0:2])
   #print "computeTestDataPredictions x:",x.shape, x
   #print "weights T:",np.asmatrix(weights).T.shape, np.asmatrix(weights).T
   #print "wTx:",np.asmatrix(weights).T.dot(x).T
   #print "sign(wTx):", np.sign(np.asmatrix(weights).T.dot(x).T)
   return np.sign(np.asmatrix(weights).T.dot(x).T)

#7.1 1 - Generate data   
data = createDataAsXYL(60)                                           #nx3 (X1,X2,C)
testdata = np.hstack(( createDataAsXYL(100/2), np.zeros(( 100,1 )) ))  #nx4 (X1,X2,C,P)

#7.1 2 - find weights minimizing the squared error
minimumweights = computeMinimumWeights(data)

#7.1 3 - Calculate predictions with the weights
testdata[:,3] = computeTestDataPredictions(testdata, minimumweights).flatten()
datac1, datac2 = getXYDataByClasses()
testdatac1p1, testdatac1p2, testdatac2p1, testdatac2p2 = getXYTestdataByClassesAndPredictions()

#7.1 4 - Calculate % correct
percent_correct_for_test = 100 * float(len(testdatac1p1) + len(testdatac2p2)) / float(len(testdata))
print percent_correct_for_test,"percent correct"


#correctly classified test data
mplt.scatter(testdatac1p1.T[0], testdatac1p1.T[1], color='blue',  marker='s', label='Test C:-1 P:-1')
mplt.scatter(testdatac1p2.T[0], testdatac1p2.T[1], color='blue',  marker=r'$\star$', label='Test C:-1 P:+1')
#incorrectly classified test data
mplt.scatter(testdatac2p1.T[0], testdatac2p1.T[1], color='red',  marker='s', label='Test C:+1 P:-1')
mplt.scatter(testdatac2p2.T[0], testdatac2p2.T[1], color='red',  marker=r'$\star$', label='Test C:+1 P:+1')
#training data
mplt.scatter(datac1.T[0], datac1.T[1], color='green',  marker='s', label='Train C:+1')
mplt.scatter(datac2.T[0], datac2.T[1], color='green',  marker=r'$\star$', label='Train C:-1')
mplt.legend(loc=1)
mplt.savefig('7.2.png', bbox_inches='tight')
