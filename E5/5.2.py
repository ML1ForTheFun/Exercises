import numpy as np
import matplotlib.pyplot as mplt
import pylab

polynominal_order = 9                                    # k
number_of_terms = polynominal_order+1                    # k+1 total terms
number_of_inputs = sum(list(range(number_of_terms+1)))   # 55 inputs for k=9, 10 for k=3




# --- load data ---
def makeTrainingData():
   #return x1, x2, y
   return np.genfromtxt('TrainingRidge.csv', delimiter=',', skip_header=1).T

def makeValidationData():
   #return x1, x2
   return np.genfromtxt('ValidationRidge.csv', delimiter=',', skip_header=1).T




#partition a data set into subsets
def partitionDataIntoNumberofparitions(data, numberofparitions):
   shuffled_data = data #prevent modification of original data
   np.random.shuffle(shuffled_data)
   
   #return [[fold*len(shuffled_data)/number,fold*len(shuffled_data)/number+len(shuffled_data)/number-1] for fold in range(number)]
   return [shuffled_data[fold*len(shuffled_data)/numberofparitions : fold*len(shuffled_data)/numberofparitions+len(shuffled_data)/numberofparitions-1] for fold in range(numberofparitions)]

#find R, our total error with regularization
def computeTotalRegularizationerror(inputarray, y, regularizationparameter):
   global number_of_inputs
   #use normal equation to find optimum weights:
   #theta = (XT.X)^-1 * XT.y
   optimum_weightarray = np.asarray(np.dot(\
                                    np.asmatrix(np.dot(inputarray.T,inputarray)).I, \
                                    np.dot(inputarray.T,y)\
                              ))
   '''
   #theta = (XT.X - lambda*I)^-1 * XT.y
   regularized_optimum_weightarray_innerterm = regularizationparameter * np.identity(number_of_inputs)
   regularized_optimum_weightarray_innerterm[0,0]=0
   optimum_weightarray = np.asarray(np.dot(\
                                    np.asmatrix(np.dot(inputarray.T,inputarray) - regularized_optimum_weightarray_innerterm).I, \
                                    np.dot(inputarray.T,y)\
                              ))
   '''
   errors = .5 * (y- np.dot(inputarray, optimum_weightarray.T))**2
   #print "inputs dot weights:\n", np.dot(inputarray, optimum_weightarray.T)
   trainingerror =  (1.0/len(inputarray))*np.sum( errors )
   regularizationerror = regularizationparameter*(.5 * np.sum(optimum_weightarray**2))
   return trainingerror + regularizationerror
   


#turn data x1, x2 into inputs.  Each x1,x2 combination produces a set of inputs with length = number_of_inputs
def computeInputs(x1array, x2array):
   return [[x1x2[0]**kx1 * x1x2[1]**kx2 for kx1 in range(number_of_terms) for kx2 in range(number_of_terms-kx1)] for x1x2 in zip(x1array, x2array) ]


#use cross-validation to find total error
def computeTotalRegularizationerrorByCrossValidation((x1, x2, y), regularizationparameter, folds):
   partitioned_data = partitionDataIntoNumberofparitions(zip(x1, x2, y), folds)
   regularizationerrors = []
   for partition in partitioned_data:
      x1p, x2p, yp = zip(*partition)
      
      regularizationerrors.append( computeTotalRegularizationerror(np.asarray(computeInputs(x1p, x2p)), yp, regularizationparameter) )
   
   return np.sum(regularizationerrors) / folds

   
def normalizeInputs(inputarray):
   norm=np.linalg.norm(inputarray)
   return inputarray if norm==0 else inputarray/norm





#5.2a
visited_regularizationparameters = []
visited_regularizationparameter_errors = []
minimum_regularizationerror = -1
minimum_regularizationparameter = 1000
for regularizationparameter in np.linspace(0, 10, 1000):
   #print regularizationparameter
   visited_regularizationparameters.append( [regularizationparameter, computeTotalRegularizationerrorByCrossValidation(makeTrainingData(), regularizationparameter, 10)] )
   print visited_regularizationparameters[-1]
   if visited_regularizationparameters[-1][1] < minimum_regularizationerror or minimum_regularizationerror==-1:
      minimum_regularizationerror = visited_regularizationparameters[-1][1]
      minimum_regularizationparameter = visited_regularizationparameters[-1][0]

#print visited_regularizationparameters
print minimum_regularizationparameter, minimum_regularizationerror



