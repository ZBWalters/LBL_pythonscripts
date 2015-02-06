from twoDplot import *
from numpy.fft import *

##load pickled files
phisvec=load("sCEPvec.pkl")
phisvec=list(phisvec)
phir1vec=load("ram1CEPvec.pkl")
phir1vec=list(phir1vec)
phir2vec=load("ram2CEPvec.pkl")
phir2vec=list(phir2vec)

nsvec=load("nsvec.pkl")
nsvec=list(nsvec)
nram1vec=load("nram1vec.pkl")
nram1vec=list(nram1vec)
nram2vec=load("nram2vec.pkl")
nram2vec=list(nram2vec)

tmeasurevec=load("tmeasurevec.pkl")
tmeasurevec=list(tmeasurevec)
woutvec=load("woutvec.pkl")
woutvec=list(woutvec)

phis_phir1_phir2_tarray=load("phis_phir1_phir2_tarray.pkl")
phis_phir1_phir2_warray=load("phis_phir1_phir2_warray.pkl")
ns_nr1_nr2_tarray=load("ns_nr1_nr2_tarray.pkl")
ns_nr1_nr2_warray=load("ns_nr1_nr2_warray.pkl")

#helper functions


def nDFFT(canglediparray,axislist, shiftaxes=False, inv=False):
    if(inv):
        fftdiparray=ifftn(canglediparray,axes=axislist)
    else:
        fftdiparray=fftn(canglediparray,axes=axislist)
    if(shiftaxes):
        if(inv):
            fftdiparray=ifftshift(fftdiparray,axes=axislist)
        else:
            fftdiparray=fftshift(fftdiparray,axes=axislist)
    return fftdiparray


def make_ns_vs_tmeasure_figs(figfun=contourplot, **kwargs):
    nsarray,woutarray=vecstoarrays(nsvec,woutvec)
    for i,j in product(range(len(nram1vec)),range(len(nram2vec))):
        figfun(nsarray, woutarray, ns_nr1_nr2_warray[:, i, j, :],
                       legend=str(nram1vec[i])+" "+str(nram2vec[j]),
                       ymultfactor=Hrt, **kwargs)

def removestreakingphase(ws=.0565):
    tmpnnntarray=copy(ns_nr1_nr2_tarray)
    (n0,n1,n2,n3)=shape(tmpnnntarray)
    for i,j in product(range(n0),range(n3)):
        n=nsvec[i]
        t=tmeasurevec[j]
        tmpnnntarray[i,:,:,j]*=exp(-1j*ws*n*t)
    return tmpnnntarray

def make_phi_n_n_qE_figs(ws=.0565, figfun=contourplot, nram1list=[nram1vec.index(1)],
                         nram2list=[nram2vec.index(-1)], **kwargs):
    tmpnnntarray=removestreakingphase(ws)
    tmpnnnwarray=nDFFT(tmpnnntarray,axislist=[3],shiftaxes=True)
    tmpphinnwarray=nDFFT(tmpnnnwarray,axislist=[0],shiftaxes=True,inv=True)
    phiarray,woutarray=vecstoarrays(phisvec,woutvec)
    for i,j in product(nram1list,nram2list):
        figfun(phiarray, woutarray, tmpphinnwarray[:,i,j,:],
                       legend=str(nram1vec[i])+" "+str(nram2vec[j]),
                       ymultfactor=Hrt, **kwargs) 
