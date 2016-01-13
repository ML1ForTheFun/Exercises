import numpy as np, matplotlib.pyplot as mplt, pylab
from matplotlib.legend_handler import HandlerLine2D

# ------- Data Setup -------
def createDataAsXYL(numberperset=60):
   dataa = [[None, None, None] for x in range(numberperset)]
   datab = [[None, None, None] for x in range(numberperset)]
   covariance_matrix = [[.5,0],[0,.5]]
   for i in range(0, 1*numberperset, 1):
      dataa[i] = np.append(np.random.multivariate_normal([.5, 1], covariance_matrix), -1)
      datab[i] = np.append(np.random.multivariate_normal([2.5, .5], covariance_matrix), 1)
      
   return (np.asarray(dataa), np.asarray(datab))

def getDataWithLabel(data, label):
   return data[data.T[2]==label]

dataa, datab = createDataAsXYL(numberperset=60)

mplt.plot([0, .5], [0,.5], color='black', label='Weight')
mplt.scatter(dataa.T[0], dataa.T[1], color='blue',  marker='s', label='Squares')        #generated as  1, marked as -1
mplt.scatter(datab.T[0], datab.T[1], color='red',  marker=r'$\star$', label='Stars') #generated as -1, marked as  1
ax = mplt.axes()
ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
#mplt.legend(handler_map={Squares: HandlerLine2D(numpoints=4)})
mplt.legend(loc=1)



mplt.savefig('7.1.png', bbox_inches='tight')
