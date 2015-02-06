from numpy import *
from scipy import *

def zbv(n,nmax):
    retvec=zeros(nmax)
    retvec[0]=-(-1)**n
    retvec[n]=1
    return retvec

def bmat(nmax):
    retmat=zeros((nmax-1,nmax))
    for n in range(1,nmax):
        zvec=zbv(n,nmax)
        for np in range(nmax):
            retmat[n-1,np]=zvec[np]
    return retmat
