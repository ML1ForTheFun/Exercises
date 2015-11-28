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
   
def getXYTraindataByClassesAndPredictions():
   c1 = traindata[traindata.T[2]==-1]
   c2 = traindata[traindata.T[2]==1]
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


perclasssample_list = [1,2,3,4,5,10,20,50]
percent_correct_for_train_means = []
percent_correct_for_test_means = []
weightx_means = []
weighty_means = []
percent_correct_for_train_stds = []
percent_correct_for_test_stds = []
weightx_stds = []
weighty_stds = []
for perclasssamplesize in perclasssample_list:
   percent_correct_for_train_list = []
   percent_correct_for_test_list = []
   weightx_list = []
   weighty_list = []

   #7.1 1 - Generate data
   testdata = np.hstack(( createDataAsXYL(1000/2), np.zeros(( 1000,1 )) ))  #nx4 (X1,X2,C,P)
   
   for trail in range(50):
      #7.1 1 - Generate data
      data = createDataAsXYL(perclasssamplesize)                               #nx3 (X1,X2,C)
      traindata = np.hstack(( data, np.zeros(( 2*perclasssamplesize,1 )) ))    #nx4 (X1,X2,C,P)

      #7.1 2 - find weights minimizing the squared error
      minimumweights = computeMinimumWeights(data)

      #7.1 3 - Calculate predictions with the weights
      traindata[:,3] = computeTestDataPredictions(data, minimumweights).flatten()
      testdata[:,3] = computeTestDataPredictions(testdata, minimumweights).flatten()
      datac1, datac2 = getXYDataByClasses()
      traindatac1p1, traindatac1p2, traindatac2p1, traindatac2p2 = getXYTraindataByClassesAndPredictions()
      testdatac1p1, testdatac1p2, testdatac2p1, testdatac2p2 = getXYTestdataByClassesAndPredictions()

      #7.1 4 - Calculate % correct
      percent_correct_for_train = 100 * float(len(traindatac1p1) + len(traindatac2p2)) / float(len(traindata))
      percent_correct_for_test = 100 * float(len(testdatac1p1) + len(testdatac2p2)) / float(len(testdata))
      #print percent_correct_for_train,"percent correct for",perclasssamplesize,"training samples"
      #print percent_correct_for_test,"percent correct for",perclasssamplesize,"test samples"

      percent_correct_for_train_list.append( percent_correct_for_train )
      percent_correct_for_test_list.append( percent_correct_for_test )
      weightx_list.append( minimumweights[0,0] )
      weighty_list.append( minimumweights[1,0] )

      
      '''
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
      mplt.savefig('7.2_N='+str(perclasssamplesize)+'.png', bbox_inches='tight')
      mplt.clf()
      '''
   '''
   print "Samplesize =",str(2*perclasssamplesize)
   print len(percent_correct_for_train_list), percent_correct_for_train_list
   print len(percent_correct_for_test_list), percent_correct_for_test_list
   print len(weightx_list), weightx_list
   print len(weighty_list), weighty_list
   '''
   
   percent_correct_for_train_means.append( np.mean(percent_correct_for_train_list) )
   percent_correct_for_test_means.append( np.mean(percent_correct_for_test_list) )
   weightx_means.append( np.mean(weightx_list) )
   weighty_means.append( np.mean(weighty_list) )
   
   percent_correct_for_train_stds.append( np.std(percent_correct_for_train_list) )
   percent_correct_for_test_stds.append( np.std(percent_correct_for_test_list) )
   weightx_stds.append( np.std(weightx_list) )
   weighty_stds.append( np.std(weighty_list) )

'''   
print percent_correct_for_train_means
print percent_correct_for_test_means
print weightx_means
print weighty_means
'''

#training data
mplt.yscale('log')
mplt.plot(perclasssample_list, percent_correct_for_train_means, color='firebrick',  label='Train % Correct mean')
mplt.plot(perclasssample_list, percent_correct_for_test_means, color='tomato',  label='Test % Correct mean')
mplt.plot(perclasssample_list, weightx_means, color='slateblue',  label='w1 mean')
mplt.plot(perclasssample_list, weighty_means, color='cornflowerblue',  label='w2 mean')
mplt.plot(perclasssample_list, percent_correct_for_train_stds, color='seagreen',  label='Train % Correct std')
mplt.plot(perclasssample_list, percent_correct_for_test_stds, color='greenyellow',  label='Test % Correct std')
mplt.plot(perclasssample_list, weightx_stds, color='indigo',  label='w1 std')
mplt.plot(perclasssample_list, weighty_stds, color='fuchsia',  label='w2 std')
mplt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
mplt.savefig('7.2.png', bbox_inches='tight')
mplt.clf()


