import numpy as np, matplotlib.pyplot as mplt, pylab

def createDataAsXYL(numberperset=60):
   data = [[0, 0, 0]]
   covariancematrix = [[.1,0,0],[0,.1,0],[0,0,0]]
   for i in range(numberperset):
      data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 1, 1], covariancematrix) + np.random.multivariate_normal([ 1, 0, 1], covariancematrix))], axis=0)
      #data = np.append(data, [.5*(np.random.multivariate_normal([ 0, 0,-1], covariancematrix) + np.random.multivariate_normal([ 1, 1,-1], covariancematrix))], axis=0)
      data = np.append(data, [.5*(np.random.multivariate_normal([ 1, 1,-1], covariancematrix) + np.random.multivariate_normal([ 1, 1,-1], covariancematrix))], axis=0)
      
   return data[1:].T

def getDataWithLabel(data, label):
   return data.T[data[2]==label].T

def findNearestNeighborIndiciesTo(data, pointindex, numberofneighbors):
   #keep record of distances and indicies of our minimums
   mindistances = [999999] * numberofneighbors
   minindicies = [pointindex] * numberofneighbors
   for otherpointindex in range(len(data)):
      #print str(pointindex)+" --- "+str(otherpointindex)+" --- "+str(data[otherpointindex])
      if otherpointindex==pointindex:
         continue
      
      #find distance to other point
      distance = computeDistance(data[pointindex][0], data[pointindex][1], data[otherpointindex][0], data[otherpointindex][0])
      #print "("+str(data[pointindex][0])+","+str(data[pointindex][1])+") - ("+str(data[otherpointindex][0])+","+str(data[otherpointindex][1])+") : "+str(distance)
      #print str(pointindex)+" - "+str(otherpointindex)+" : "+str(distance)

      #find the max of our minimum points
      maxmindistance = -1
      maxminindex = 0
      for minpointindex in range(len(minindicies)):
         #print "\t\tmpi:"+str(minpointindex)+" indicies:"+str(minindicies)+" distances:"+str(mindistances)
         if minindicies[minpointindex] == pointindex:
            maxminindex = minpointindex
            maxmindistance = mindistances[minpointindex]
            #print "\t\tsame point"
            break
         #print("\t\tmin["+str(minpointindex)+"]:"+str(mindistances[minpointindex])+ " >? min["+str(maxminindex)+"]:"+str(mindistances[maxminindex]))
         if mindistances[minpointindex] > mindistances[maxminindex]:
            maxminindex = minpointindex
            maxmindistance = mindistances[minpointindex]
            #print "\t\t\tnew maxmin min["+str(maxminindex)+"]:"+str(mindistances[maxminindex])
            
      #print "\tmin["+str(maxminindex)+"]:"+str(mindistances[maxminindex])+" minindicies:"+str(minindicies)+" mindistances:"+str(mindistances)

      #replace max with this new value if closer
      #print "\tnewdistance: "+str(distance)+" <? olddistance: "+str(mindistances[maxminindex])
      if distance < mindistances[maxminindex]:
         mindistances[maxminindex] = distance
         minindicies[maxminindex] = otherpointindex
         #print "\t\tadded to minindicies["+str(maxminindex)+"]:"+str(otherpointindex)+" mindistances["+str(maxminindex)+"]:"+str(distance)
      
   return minindicies

def computeDistance(ax,ay,bx,by):
   return np.sqrt((ax-bx)**2 + (ay-by)**2)
   
   
#6.1
data = createDataAsXYL(30)

#6.1a
centerpointindex = 7
nearestneighborindicies = findNearestNeighborIndiciesTo(data.T, centerpointindex, 3)
print centerpointindex, nearestneighborindicies

#6.1b
mplt.scatter(getDataWithLabel(data,  1)[0], getDataWithLabel(data,  1)[1], color='blue', marker='s')
mplt.scatter(getDataWithLabel(data, -1)[0], getDataWithLabel(data, -1)[1], color='blue', marker=r'$\star$')
mplt.savefig('6.1b.png', bbox_inches='tight')