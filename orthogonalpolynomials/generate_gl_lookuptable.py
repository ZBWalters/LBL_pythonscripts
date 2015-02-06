from scipy.special import *

def printarray(name,array):
    xstr="static double "+name+str(order)+"["+str(order)+"]={"
    for i in range(len(array)):
        xstr=xstr+ "%18.15f" % array[i]+", "
    xstr=xstr[:-2]+"};"
    print xstr

nmax=50
for order in range(1,nmax+1):
    [legpts,legweights]=orthogonal.p_roots(order)
    printarray('x',legpts)
    printarray('w',legweights)
    print ""

#for order in range(1,nmax+1):
#    print "{ x"+str(order)+", w"+str(order)+"},"

outstr="{"
for order in range(1,nmax+1):
    outstr=outstr+"w"+str(order)+", "
outstr=outstr[:-2]+"};"
print outstr
