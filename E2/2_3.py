import matplotlib.pyplot as mplt, numpy, pylab

xs = numpy.linspace(-2, 2, 100)
maxhiddenunits = 10
maxmlps = 50
bestfunctions = []
for std in [0.5,2]:
    mse = []
    functions = []
    for mlp in range(maxmlps):
        yvaluesoverx = [0 for x in xs]
        for i in range(maxhiddenunits):
            ai = numpy.random.normal(0,std)
            bi = numpy.random.uniform(-2,2)
            wi = numpy.random.normal(0,1)
            yvaluesoverx += wi * numpy.tanh(ai*(xs - bi))
        mplt.plot(xs, yvaluesoverx)
        mse.append(numpy.mean((yvaluesoverx + xs)**2))
        functions.append(yvaluesoverx)
    pylab.savefig('./2_3_std={0}.png'.format(std), bbox_inches='tight')
    mplt.clf()
    bestfunctions.append(functions[mse.index(min(mse))])

for f in bestfunctions:
    mplt.plot(xs,f)
pylab.savefig('./2_3_star.png', bbox_inches='tight')
mplt.clf()