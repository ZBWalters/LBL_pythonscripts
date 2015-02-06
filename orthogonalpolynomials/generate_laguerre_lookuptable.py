from numpy import *
from scipy import *
from numpy.polynomial.laguerre import lagroots

def laguerreseries(order):
    retarray=zeros(order+1)
    retarray[-1]=1.
    return retarray

def laguerreroots(order):
    return lagroots(laguerreseries(order))

def lagval(order,x):
    if(order==0):
        return 1
    if(order==1):
        return 1.-x
    orderminus1val=lagval(order-1,x)
    orderminus2val=lagval(order-2,x)
    k=order-1
    return ((2.*k+1.-x)*orderminus1val-k*orderminus2val)/(k+1.)


def laguerreweights(order):
    roots=laguerreroots(order)
    weights=zeros(order)
    lnp1vals=zeros(order)
    for i in range(order):
        lnp1vals[i]=lagval(order+1,roots[i])
    for i in range(order):
        weights[i]=roots[i]/(((order+1.)**2)*lnp1vals[i]**2.)
    return weights

#routines for printing arrays
def printarray(name,array):
    xstr="static double "+name+str(order)+"["+str(order)+"]={"
    for i in range(len(array)):
        #xstr=xstr+ "%18.15f" % array[i]+", "
        xstr=xstr+ "%18.15e" % array[i]+", "
    xstr=xstr[:-2]+"};"
    print xstr


nmax=50
for order in range(1,nmax+1):
    points=laguerreroots(order)
    weights=laguerreweights(order)
    printarray('x',points)
    printarray('w',weights)
    print ""

outstr="{"
for order in range(1,nmax+1):
    outstr=outstr+"w"+str(order)+", "
outstr=outstr[:-2]+"};"
print outstr
