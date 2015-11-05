def computeErrorFromMLPWithHiddenunitsAndData(number_of_hiddenunits, x, t):
   import numpy as np
   import matplotlib.pyplot as mplt
   import pylab
   
   #Training data
   #(x, t) = np.genfromtxt('RegressionData.txt').T;
   p = len(x)
   nhu = number_of_hiddenunits
   x = x.reshape(p,1)
   t = t.reshape(p,1)
   
   #Weights and biases are set randomly in the range [-0.5, 0.5);
   w = np.asarray([0.5 * np.random.randn(nhu), 0.5 * np.random.randn(nhu)]);
   b = np.asarray([0.5 * np.random.randn(nhu), 0.5 * np.random.randn(1)]);
   #b = np.asarray([[1.,1.,1.],[1.]])
   
   a = np.tanh(np.dot(x,w[0].reshape(1,nhu))-b[0]);
   o = (np.mean(a*np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu)),axis=1) - b[1]*np.ones(p)).reshape(p,1);
   
   MeanErrors = [0]
   
   for i in range(3000):
      #Error cost function
      e = 0.5*(o-t)**2;
      
      MeanErrors.append(np.mean(e))
      
      #break criterium
      if np.abs(MeanErrors[-1]-MeanErrors[-2])/MeanErrors[-1]<10**-5: 
          break
      
      #Delta_j^v for calculating the weight shift; delta for output is 1 for the identity transfer function
      d = 4*np.cosh(np.dot(x,w[0].reshape(1,nhu)))**2/np.cosh(2*np.dot(x,w[0].reshape(1,nhu))+1)**2*np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu))
      
      #weight adjustion
      dw1 = np.mean((o-t)*a,axis=0)
      dw0 = np.mean((o-t)*d*x,axis=0)
      
      w[0]-=0.05*dw0
      w[1]-=0.05*dw1
      
      a = np.tanh(np.dot(x,w[0].reshape(1,nhu))-b[0]);
      o = (np.mean(a*np.dot(np.ones(p).reshape(p,1),w[1].reshape(1,nhu)),axis=1) - b[1]*np.ones(p)).reshape(p,1);
      
   x = np.linspace(0.,1.,100).reshape(100,1)
   a = np.tanh(np.dot(x,w[0].reshape(1,nhu))-b[0]);
   o = (np.mean(a*np.dot(np.ones(100).reshape(100,1),w[1].reshape(1,nhu)),axis=1) - b[1]*np.ones(100)).reshape(100,1);
   
   return (np.sum(e), x, a, o)
   


import matplotlib.pyplot as mplt, numpy, pylab, sys

#3.3a
#data
x, fx = zip(*[(x, numpy.sin(1.5*numpy.pi*x)+numpy.random.normal(0,.8)) for x in numpy.linspace(0, 1, 100)])

#3.3b
#indices for a randomized x, f_x order
shuffled_indicies = range(0,len(x))
numpy.random.shuffle(shuffled_indicies)
#print shuffled_indicies

#get error for each network with varying number of hidden units
number_of_folds = 5
fold_size = len(x)/number_of_folds
best_number_of_hiddenunits = -1
smallest_total_hiddenunit_error = sys.maxsize
for number_of_hiddenunits in range(1,11):
   total_hiddenunit_error = 0
   for fold in range(0,number_of_folds):
      #generate a subset of the data
      subset_indicies = shuffled_indicies[fold*fold_size:fold*fold_size+fold_size-1]   #indices for this subset
      subset_x = numpy.array(x)[subset_indicies]
      subset_fx = numpy.array(fx)[subset_indicies]
      
      #find error
      (fold_error, plotx, plota, ploto) = computeErrorFromMLPWithHiddenunitsAndData(number_of_hiddenunits, subset_x, subset_fx)
      total_hiddenunit_error = total_hiddenunit_error + fold_error
      
      print "\t"+str(number_of_hiddenunits)+"/"+str(fold)+" error: "+str(fold_error)+"/"+str(total_hiddenunit_error)
      
   total_hiddenunit_error = (1.0 / len(x))*total_hiddenunit_error
   print str(number_of_hiddenunits)+" "+str(total_hiddenunit_error)
   
   #manage minium error
   if(total_hiddenunit_error < smallest_total_hiddenunit_error):
      smallest_total_hiddenunit_error = total_hiddenunit_error
      best_number_of_hiddenunits = number_of_hiddenunits

print "bests: "+str(best_number_of_hiddenunits)+" "+str(smallest_total_hiddenunit_error)

#3.3c
(xrg, fxrg) = numpy.genfromtxt('RegressionData.txt').T
for xrgi in xrg:
   x = numpy.append(x, xrgi)
for fxrgi in fxrg:
   fx = numpy.append(fx, fxrgi)
(error, plotx, plota, ploto) = computeErrorFromMLPWithHiddenunitsAndData(best_number_of_hiddenunits, numpy.array(x), numpy.array(fx))
#plot
mplt.scatter(x, fx, color='darkblue')
mplt.scatter(xrg, fxrg, color='skyblue')

mplt.plot(plotx,ploto)

pylab.savefig('3.3c.png', bbox_inches='tight')
mplt.clf()

