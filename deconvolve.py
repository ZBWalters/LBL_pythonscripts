import sys
sys.path.append(r'/Users/zwalters/pythonscripts')
from makefigs import *



def fouriermat(ftinpvec):
    nvec=len(ftinpvec)
    retmat=zeros((nvec,nvec))*0j
    for j in range(nvec):
        retmat[:,j]=roll(ftinpvec,-int(floor(nvec/2))+j)
#    for i,j in product(range(nvec),range(nvec)):
##        k=i-j
##        if((k>=-nvec/2) and (k<nvec/2)):
##            retmat[i,j]=ftinpvec[nvec/2+k]
#        retmat[mod(i-nvec/2+j,nvec),j]=ftinpvec[i]
    return retmat


def pulse(t, wpr, wosc=.0565, nosc=0, ncycle=8, envpow=2):
    tmax=ncycle*2*pi/wosc
    wenv=wosc/(2*ncycle)
    retval=0j
    if(t<=tmax):
        retval=exp(1j*(wpr+nosc*wosc)*t)*pow(sin(wenv*t),envpow)
    return retval

def pulsevec(wpr,wosc=.0565, nosc=0, ncycle=8, envpow=2):
    tmax=ncycle*2*pi/wosc
    dt=tvec[1]-tvec[0]
    ntmat=int(ceil(tmax/dt))
    tmptvec=arange(0,tmax,dt)
    retvec=zeros(len(tvec))*0j
    for i in range(len(tvec)):
        retvec[i]=pulse(tvec[i], wpr, wosc, nosc, ncycle, envpow)
    return retvec

def deconvolve(dipvec,sourcevec,tiny=.1):
    ftdipole=fft(dipvec)
    ftdipole/=abs(ftdipole).max()
    ftsource=fft(sourcevec)
    ftsource/=abs(ftsource).max()
    np=len(ftdipole)
    ftresp=ftdipole/(ftsource+tiny*ones(np))
    resp=(ifft(fftshift(ftresp)))
    return resp

dirstr="1/"
narray,tarray,diparray=arraysfromfile(dirstr+"kIr_vs_t.dat")
tvec=tarray[0,:]
nvec=narray[:,0]

(n0,n1)=shape(diparray)
dipvec=diparray[n0/2,:]
