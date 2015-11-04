def computeErrorFromMLPWithHiddenunitsAndData(number_of_hiddenunits, x, t):
   #Initialization
   import numpy as np
   
   #Training data
   #(x, t) = np.genfromtxt('RegressionData.txt').T;

   #Number of neurons per layer.
   n = [1, number_of_hiddenunits, 1];

   #Weights and biases are set randomly in the range [-0.5, 0.5);
   w = [-.5 + np.random.randn(i, j) for i, j in zip(n[:-1], n[1:])];
   b = [np.random.randn(i) - .5 for i in n[1:]];

   #Forward propagation
   #activation of neurons, layer by layer
   a = np.squeeze(np.tanh([np.dot(w[0], i) - b[0] for i in x]));
   o = np.squeeze(np.asarray([np.dot(w[1].T, o.T) - b[1] for o in a]));

   #Error cost function
   e = np.asarray([(y_t - y_w)**2/2 for y_t, y_w in zip(t, o)]);
   
   return np.sum(e)
   


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
      fold_error = computeErrorFromMLPWithHiddenunitsAndData(number_of_hiddenunits, subset_x, subset_fx)
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
#plot
mplt.scatter(x, fx, color='darkblue')
(xa, fxa) = numpy.genfromtxt('RegressionData.txt').T
mplt.scatter(xa, fxa, color='skyblue')
pylab.savefig('3.3c.png', bbox_inches='tight')
mplt.clf()

