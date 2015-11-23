import numpy as np, matplotlib.pyplot as mplt, pylab

def createDataAsXYL(numberperset=60):
   data = [[0, 0, 0]]
   covariancematrix = [[.1,0,0],[0,.1,0],[0,0,0]]
   for i in range(numberperset):
      data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 1, 1], covariancematrix) + np.random.multivariate_normal([ 1, 0, 1], covariancematrix))], axis=0)
      #data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 0,-1], covariancematrix) + np.random.multivariate_normal([ 1, 1,-1], covariancematrix))], axis=0)
      data = np.append(data, [.5*(np.random.multivariate_normal([ 1, 1,-1], covariancematrix) + np.random.multivariate_normal([ 1, 1,-1], covariancematrix))], axis=0)
      
   return data[1:]

def getDataWithLabel(data, label):
   return data[data.T[2]==label]

def findNearestNeighborIndiciesTo(data, pointindex, numberofneighbors):
   #keep record of distances and indices of our minimums
   mindistances = [999999] * numberofneighbors
   minindices = [pointindex] * numberofneighbors
   for otherpointindex in range(len(data)):
      #print str(pointindex)+" --- "+str(otherpointindex)+" --- "+str(data[otherpointindex])
      if otherpointindex==pointindex:
         continue
      
      #find distance to other point
      distance = computeDistance(data[pointindex][0], data[pointindex][1], data[otherpointindex][0], data[otherpointindex][1])
      #print ""+str(pointindex)+" ("+str(data[pointindex][0])+","+str(data[pointindex][1])+") -> "+str(otherpointindex)+" ("+str(data[otherpointindex][0])+","+str(data[otherpointindex][1])+") : "+str(distance)

      #find the max of our minimum points
      maxmindistance = -1
      maxminindex = 0
      for minpointindex in range(len(minindices)):
         #print "\t\tmpi:"+str(minpointindex)+" indicies:"+str(minindices)+" distances:"+str(mindistances)
         #print("\t\tmin["+str(minpointindex)+"]:"+str(mindistances[minpointindex])+ " >? min["+str(maxminindex)+"]:"+str(mindistances[maxminindex]))
         if mindistances[minpointindex] > mindistances[maxminindex]:
            maxminindex = minpointindex
            maxmindistance = mindistances[minpointindex]
            #print "\t\t\tnew maxmin min["+str(maxminindex)+"]:"+str(mindistances[maxminindex])
            
      #print "\tmin["+str(maxminindex)+"]:"+str(mindistances[maxminindex])+" minindices:"+str(minindices)+" mindistances:"+str(mindistances)

      #replace max with this new value if closer
      #print "\tnewdistance: "+str(distance)+" <? olddistance: "+str(mindistances[maxminindex])
      if distance < mindistances[maxminindex]:
         mindistances[maxminindex] = distance
         minindices[maxminindex] = otherpointindex
         #print "\t\tadded to minindices["+str(maxminindex)+"]:"+str(otherpointindex)+" mindistances["+str(maxminindex)+"]:"+str(distance)
      
   return minindices

def computeDistance(ax, ay, bx, by):
   return np.sqrt((ax-bx)**2 + (ay-by)**2)
   
   
#6.1
data = createDataAsXYL(60)
#print data

for k in [1, 3, 5]:
   classifiedasones = [None] * len(data)
   
   #find the classifications
   for centerpointindex in range(len(data)):
      classifications = data[ findNearestNeighborIndiciesTo(data, centerpointindex, k) ].T[2]
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
