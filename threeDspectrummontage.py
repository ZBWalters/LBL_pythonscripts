from numpy import *
import os
import sys
import subprocess
sys.path.append(os.path.expanduser("~/pythonscripts/"))
from twoDplot import *
from itertools import product, chain
from numpy.fft import fftn,ifftn,fftshift



#read in arrays
wphiwarray=[]
wphiwarray=load("freq_vs_phase_vs_freq.pkl")
wphitarray=[]
wphitarray=load("freq_vs_phase_vs_t.pkl")
wkIRwarray=[]
wkIRwarray=load("freq_vs_kIR_vs_freq.pkl")
wkIRtarray=[]
wkIRtarray=load("freq_vs_kIR_vs_t.pkl")
kIRvec=[]
kIRvec=load("kIRvec.pkl")
phasevec=[]
phasevec=load("phasevec.pkl")
winpvec=[]
winpvec=load("winpvec.pkl")
woutvec=[]
woutvec=load("woutvec.pkl")
tvec=[]
tvec=load("tvec.pkl")

####################################################################
def subtractdrivingfreqs(wosc=.0565):
    retarray=copy(wkIRtarray)
    (n0,n1,n2)=shape(retarray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(-1j*(winpvec[i]+kIRvec[j]*wosc)*tvec[k])
    return retarray

def subtractprobefreqs():
    retarray=copy(wkIRtarray)
    (n0,n1,n2)=shape(retarray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(-1j*(winpvec[i])*tvec[k])
    return retarray

def deltaw_vs_w(wosc=.0565):
    retarray=subtractdrivingfreqs(wosc)
    retarray=fftn(retarray,axes=[2])
    retarray=fftshift(retarray,axes=[2])
    return retarray

def iftarray(inparray,axes,shift=True):
    if(shift):
        retarray=fftshift(inparray,axes=axes)
    else:
        retarray=copy(inparray)
    retarray=ifftn(retarray,axes=axes)
    return retarray

def subtractdwphase(inparray,wosc=.0565):
    retarray=copy(inparray)
    (n0,n1,n2)=shape(inparray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(1j*phasevec[j]/wosc*woutvec[j])
    return retarray
    

def ftarray(inparray,axes,shift=True):
    retarray=fftn(inparray,axes=axes)
    if(shift):
        retarray=fftshift(retarray,axes=axes)
    return retarray

def wphiw_phasecorrected(wosc=.0565):
    wkIRwarray

####################################################################
def arraymontage(inparray, indxlist=kIRvec, montagetitle="montage.png",
                 colorpower='log', logrange=6, arrayfun=None,
                 xvec=winpvec, yvec=woutvec, removefigs=True,
                 nrows=None, nmin=None, nmax=None, nstride=1,  ylo=-10, yhi=10,
                 xlabel="$\omega_{in}$ (eV)", ylabel="$\omega_{out}$ (eV)",
                 arraynormalize=True):
    (n0,n1,n2)=shape(inparray)
    if(arraynormalize):
        zmax=abs(inparray).max()
    else:
        zmax=None
    fignamelist=[]

    xarray,yarray=vecstoarrays(xvec,yvec)

    loopnmin,loopnmax=indxrange(indxlist,nmin,nmax)
    for n in range(loopnmin,loopnmax,nstride):
        if(arrayfun==None):
            fign=imshowplot_hsv(xarray,yarray,inparray[:,n,:],
                                colorpower=colorpower,logrange=logrange,
                                legend="$n_{streak}=$"+indxstr(n,indxlist),
                                ymultfactor=Hrt, xmultfactor=Hrt, absmax=zmax,
                                xlabel=xlabel, ylabel=ylabel,ylo=ylo,yhi=yhi)
        else:
            fign=imshowplot_fun(xarray,yarray,inparray[:,n,:],
                                colorpower=colorpower,logrange=logrange,
                                legend="$n_{streak}=$"+indxstr(n,indxlist),
                                ymultfactor=Hrt, xmultfactor=Hrt,
                                arrayfunction=arrayfun, absmax=zmax,
                                xlabel=xlabel, ylabel=ylabel,ylo=ylo,yhi=yhi)
        figname="fig"+"%03d" % n+".png"
        fign.savefig(figname)
        plt.close(fign)
        fignamelist.append(figname)
    makemontage(fignamelist,nrows,montagetitle)
    if(removefigs):
        deletefiglist(fignamelist)

def indxstr(n,indxlist):
    if(indxlist==None):
        return str(n)
    else:
        return str(int(indxlist[n]))

def indxrange(indxlist,nmin=None,nmax=None):
    retnmin=0
    retnmax=len(indxlist)
    if(indxlist!=None):
        if(nmin!=None):
            retnmin=list(indxlist).index(nmin)
        if(nmax!=None):
            retnmax=list(indxlist).index(nmax)+1
    print("retnmin,retnmax\t"+str(retnmin)+", "+str(retnmax))
    return retnmin,retnmax

def makemontage(fignamelist,nmin=None,nmax=None,nrows=None,outfigname="montage.png"):
    nfigs=len(fignamelist)
    fignamestr=" ".join(fignamelist)
    if(nrows==None):
        nrows=int(floor(sqrt(nfigs)))
    ncols=int(ceil(nfigs/nrows))
    montagecommand="montage -tile "+str(nrows)+"x"+str(ncols)+ " -mode Concatenate "+fignamestr+" "+outfigname
    subprocess.call(montagecommand.split())

def deletefiglist(fignamelist):
    for figname in fignamelist:
        os.remove(figname)


def vecstoarrays(vec1,vec2):
    n=len(vec1)
    m=len(vec2)
    retarray1=outer(vec1,ones(m))
    retarray2=outer(ones(n),vec2)
    return retarray1,retarray2

 
