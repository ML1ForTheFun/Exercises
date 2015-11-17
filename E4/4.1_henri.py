import numpy, random
import matplotlib.pyplot as mplt, pylab

ERROR_THRESHOLD = 10**-5

# === helper functions ===
def getH(X):
   return numpy.dot(X, X.T)
   
def getGradient(X, t, w):
   b = -numpy.dot(X, t.T)
   
   return numpy.dot(getH(X), w) + b

def getError(X, t, w):
   difference = numpy.dot(w.T, X) - t
   return .5*difference*difference.T

def checkIfBreakConditionMetFromErrors(current_error, new_error):
   global ERROR_THRESHOLD
   return numpy.abs(new_error-current_error)/new_error<ERROR_THRESHOLD
   
def initialize():
   x = numpy.array([-1 , .3, 2])
   t = numpy.array([-.1, .5, .5])
   X = numpy.array([[1, 1, 1], [-1 , .3, 2]])
   w = numpy.array([random.uniform(-2,2), random.uniform(-2,2)]).T

   return x, X, t, w


# === Searches ===
def updateWeightsByRecursiveGradientDescent(X, t, w, eta, depth, visited_weights, current_error):
   new_w = w - eta*getGradient(X, t, w)
   new_error = numpy.mean(getError(X, t, new_w))
   
   if checkIfBreakConditionMetFromErrors(current_error, new_error) or depth > 1200:
      print "Gradient Descent ended at: "+str(depth)
      return visited_weights
   else:
      visited_weights.append(new_w)
      return updateWeightsByRecursiveGradientDescent(X, t, new_w, eta, depth+1, visited_weights, new_error)


def updateWeightsByRecursiveLineSearch(X, t, w, depth, visited_weights, current_error):
   g = getGradient(X, t, w)
   alpha = - numpy.dot(g.T, g) / numpy.dot(  g.T, numpy.dot(getH(X), g)  )
   new_w = w + alpha*g
   new_error = numpy.mean(getError(X, t, new_w))
   
   if checkIfBreakConditionMetFromErrors(current_error, new_error) or depth > 1200:
      print "Line Search ended at: "+str(depth)
      return visited_weights
   else:
      visited_weights.append(new_w)
      return updateWeightsByRecursiveLineSearch(X, t, new_w, depth+1, visited_weights, new_error)
   
      

def updateWeightsByRecursiveConjugateGradient(X, t, w, g, d, depth, visited_weights, current_error):
   H = getH(X)
   alpha = - numpy.dot(d.T, g) / numpy.dot( d.T, numpy.dot(H, d) )
   new_w = w + alpha*d
   new_error = numpy.mean(getError(X, t, new_w))
   
   if checkIfBreakConditionMetFromErrors(current_error, new_error) or depth > 1200:
      print "Recursive Conjugate Gradient ended at: "+str(depth)
      return visited_weights
   else:
      b = -numpy.dot(X, t.T)
      new_g = numpy.dot(H, new_w) + b
      beta = numpy.dot(new_g.T, new_g) / numpy.dot(g.T, g)
      new_d = new_g + beta*d
      visited_weights.append(new_w)
      return updateWeightsByRecursiveConjugateGradient(X, t, new_w, new_g, new_d, depth+1, visited_weights, new_error)
   
   
# === plotting ===
x, X, t, w = initialize()

current_error = numpy.mean(getError(X, t, w))
g = getGradient(X, t, w)
gradientdescent_weights = updateWeightsByRecursiveGradientDescent(X, t, w, .01, 0, [w], current_error)
linesearch_weights = updateWeightsByRecursiveLineSearch(X, t, w, 0, [w], current_error)
conjugategradient_weights = updateWeightsByRecursiveConjugateGradient(X, t, -g, g, -g, 0, [-g], current_error)

mplt.plot([w[0] for w in gradientdescent_weights], [w[1] for w in gradientdescent_weights], color='skyblue')
mplt.plot([w[0] for w in linesearch_weights], [w[1] for w in linesearch_weights], color='maroon')
mplt.plot([w[0] for w in conjugategradient_weights], [w[1] for w in conjugategradient_weights], color='forestgreen')

pylab.savefig('4.1d.png', bbox_inches='tight')
mplt.clf()

   
