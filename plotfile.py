from numpy import *
from pylab import *
import glob


def plotfile(filename):
    array1=genfromtxt(filename)

    clrs=['black','red','green','blue','orange','brown','grey','violet']
    names=['x','real','imaginary']

    (n1,n2)=shape(array1)
    for i in range(1,n2):
        plot(array1[:,0],array1[:,i],ls="-",c=clrs[i-1],label=str(i))#label=names[i])
    xlabel('x')
    legend(loc='best')
    show()
    return

