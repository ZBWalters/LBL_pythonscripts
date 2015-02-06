from numpy import *
from numpy.linalg import eig, eigh
from numpy.fft import *
from numpy.polynomial.legendre import leggauss
from itertools import product as iterproduct
import matplotlib.pyplot as plt
from numba import jit, autojit
import sys
sys.path.append(r'/Users/zwalters/pythonscripts')
from twoDplot import *
Hrt=27.21
aut=24.2
Iat=3.51e16#atomic unit of intensity

#def envelope(t,wenv):
#    if(abs(wenv*t)<pi/2):
#        retval=pow(cos(wenv*t),2)
#    else:
#        retval=0
#    return retval

def envelope(t,wenv):
    return where(abs(wenv*t)<pi/2,pow(cos(wenv*t),2),0)

def Efield(t):
    return F0*envelope(t,wenv0)*cos(w0*t)

def twopulses(t,F0,w0,tau0,tc0,F1,w1,tau1,tc1):
    wenv0=(pi/2)/tau0
    wenv1=(pi/2)/tau1
    retval= (F0*envelope((t-tc0),wenv0)*cos(w0*(t-tc0)) +
             F1*envelope((t-tc1),wenv1)*cos(w1*(t-tc1)))
    return retval

################################################################################
#excitation amplitudes
def goldenrule(wosc,En,tau):
    dE=wosc-En
    return (2/dE)*sin(dE*tau)

def goldenruleamps(diparray,enarray,wosc,tau):
    retarray=zeros(shape(enarray))
    (nt,ns)=shape(enarray)
    for i,j in iterproduct(range(nt), range(ns)):
        retarray[i,j]=diparray[i,j]*goldenrule(wosc,enarray[i,j],tau)
    return retarray

#@autojit
#def rotwave(t,tau,wosc,Eoft,phiCEP=0):
#    nglpts=10
#    pts,wts=leggauss(nglpts)
#    dtpts=pts*tau*(pi/2)
#    dtwts=pts*tau*(pi/2)
#    vals=list(map(lambda dt: exp(-1j*((wosc-Eoft)*(t+dt))+phiCEP)*pow(cos(dt/tau),2),dtpts))
#    return sum(vals*dtwts)
#
#@autojit
#def rotwaveamps(t, tau, phiCEP, diparray, enarray, wosc):
#    retarray=zeros(shape(enarray),dtype='complex')
#    (nt,ns)=shape(enarray)
#    for i,j in iterproduct(range(nt),range(ns)):
#        retarray[i,j]=rotwave(t,tau,wosc,enarray[i,j],phiCEP)*exp(1j*w1*tarray[i])
#    return retarray
#

def ionintpts(tion,w,wenv,nglpts=5):
    #return set of time points & gauss legendre weights required for integrating
    #over a pulse
    pts,wts=leggauss(nglpts)
    pts=(pts+1)/2#map to range [0:1]
    wts=wts/2

    period=(2*pi)/w
    envperiod=(pi)/wenv
    #print("period\t"+str(period))
    #print("envperiod\t"+str(envperiod))
    ncycles=int(ceil(envperiod/period))
    tarray=zeros(nglpts*ncycles)
    wtarray=zeros(nglpts*ncycles)
    tstart=tion-ncycles/2*period
    for i,n in iterproduct(range(nglpts), range(ncycles)):
       tarray[i+n*nglpts]=tstart+(n+pts[i])*period 
       wtarray[i+n*nglpts]=wts[i]*period
    return tarray,wtarray

def ionamps(F0,pulseCEP,Envec,phasevec,evecarray,tion,w,wenv,nglpts=5):
    intpts,intwts=ionintpts(tion,w,wenv,nglpts)
    #print("intpts\t"+str(intpts))
    #print("intwts\t"+str(intwts))
    nintpts=len(intpts)
    namps=len(Envec)
    retarray=zeros(namps,dtype='complex')
    for i in range(nintpts):
        t=intpts[i]
        pulseval=exp(-1j*w*(t-tion))*envelope(t-tion,wenv)#cos(w*(t-tion))*envelope(t-tion,wenv)
        dipvec=dipg_floq(Hdip,nmin,nmax,0,nconserved=True)*exp(-1j*Envec*(t-tion))#dipole matrix elts with ground state
        dipmatels=dot(evecarray,conjugate(dipvec))
        retarray+=dipmatels*conjugate(pulseval)*intwts[i]
    #print("retarray after integration\t"+str(abs(retarray)))
#    dipvec=dipg_floq(Hdip,nmin,nmax)#dipole matrix elts with ground state
#    dipmatels=dot(evecarray,dipvec)
#    retarray*=dipmatels
    #print("shape phasevec\t"+str(shape(phasevec)))
    #retarray*=exp(1j*phasevec)#*F0
    retarray*=1j*F0#*dot(evecarray,dipg_floq(Hdip,nmin,nmax,0,nconserved=True))
    return retarray

def twodphasemat(tarray, phasemat, w):
    expphases=exp(-1j*phasemat)
    nvec=zeros(ntot, dtype='complex')
    for i in range(nstates):
        for n in range(nmin,nmax+1):
            nvec[psinindex(i,n,nmin)]=n
    nphasemat=outer(tarray,nvec*w)
    nphasemat=exp(-1j*nphasemat)
    return expphases*nphasemat

def timeordermat(n):
    onevec=ones(n)
    rnge=range(n)
    mat0=outer(rnge,onevec)
    mat1=outer(onevec,rnge)
    tomat=where(mat1>=mat0,1,0)
    return tomat


def deltatmat(tarray):
    onevec=ones(len(tarray))
    mat0=outer(tarray,onevec)
    mat1=outer(onevec,tarray)
    dtmat=mat1-mat0
    return dtmat

def accumulatedphase(phasearray,stateindx):
    #accumulated phase between time of ionization and measurement (no time ordering)
    return outer(exp(-1j*phasearray[:,stateindx]),exp(1j*phasearray[:,stateindx]))

def induceddipoles(tarray, phasearray, ionamparray,adiabaticdiparray,maskvec=None):
    if(maskvec==None):
        maskvec=ones(ntot)
    print("maskvec\t"+str(maskvec))
    twoDionvec=exp(-1j*phasearray)*(ionamparray*maskvec)
    twoDdipvec=adiabaticdiparray#exp(-1j*phasearray)*adiabaticdiparray
    retarray=dot(twoDionvec,conjugate(transpose(twoDdipvec)))
    retarray*=timeordermat(nt)
    retarray=retarray+conjugate(retarray)
    return retarray


def ionamparray(F0,pulseCEP,enarray,phasearray,evecarray,tarray,w,wenv,nglpts=5):
    retarray=zeros(shape(enarray),dtype='complex')
    for i in range(len(tarray)):
        retarray[i,:]=ionamps(F0,pulseCEP,enarray[i,:], phasearray[i,:], evecarray[i,:,:],
                              tarray[i], w, wenv, nglpts)
    return retarray

def ionamparray_fieldfreeprojection(evecarray, projvec):
    return dot(evecarray,projvec)

def ionamparray_test(evecarray,iffstate=0, nffstate=0, iffname=None):
    if(iffname!=None):
        iffstate=namedict[iffname]
    tmpvec=zeros(ntot)
    tmpvec[psinindex(iffstate,nffstate,nmin)]=1
    return dot(evecarray,tmpvec)

def fieldfree_ionamparray(ionamparray,evecarray):
    (nt,ns)=shape(ionamparray)
    retarray=zeros(shape(ionamparray),dtype='complex')
    for i in range(nt):
        retarray[i,:]=dot(ionamparray[i,:],evecarray[i,:,:])
    return retarray



#
################################################################################
#instantaneous eigenvalues/vectors of field-on Hamiltonian
@autojit
def psinindex(i,nph,nphmin):
    tmpn=nph-nphmin
    return i+tmpn*nstates

@autojit
def psinindexinv(indx,nphmin):
    retis=mod(indx,nstates)
    tmpn=int(floor(indx/nstates))
    retn=tmpn+nphmin
    return retis,retn


@autojit
def Hfloq(H0,dip,Eenv,phaseE,nmin,nmax):
    ntot=numn*nstates
    bigH=zeros((ntot,ntot),dtype='complex')
    for i,n in iterproduct(range(nstates),range(nmin,nmax+1)):
        tmpindx=psinindex(i,n,nmin)
        bigH[tmpindx,tmpindx]=H0[i,i]+n*w0

    #dipole operators corresponding to lowering photon number
    for i,j,n in iterproduct(range(nstates),range(nstates),range(nmin+1,nmax+1)):
        tmpindx0=psinindex(i,n,nmin)
        tmpindx1=psinindex(j,n-1,nmin)
        bigH[tmpindx0,tmpindx1]=dip[i,j]*Eenv*exp(1j*phaseE)
        bigH[tmpindx1,tmpindx0]=bigH[tmpindx0,tmpindx1]
    #dipole operators corresponding to raising photon number
    for i,j,n in iterproduct(range(nstates),range(nstates),range(nmin,nmax)):
        tmpindx0=psinindex(i,n,nmin)
        tmpindx1=psinindex(j,n+1,nmin)
        bigH[tmpindx0,tmpindx1]=dip[i,j]*Eenv*exp(-1j*phaseE)
        bigH[tmpindx1,tmpindx0]=bigH[tmpindx0,tmpindx1]
    return bigH

    #dipole operators corresponding to lowering photon number
    for i,j,n in iterproduct(range(nstates),range(nstates),range(nmin+1,nmax+1)):
        tmpindx0=psinindex(i,n,nmin)
        tmpindx1=psinindex(j,n-1,nmin)
        bigH[tmpindx0,tmpindx1]=dip[i,j]*Eoft
    return bigH

def dipg_floq(dip,nmin,nmax,floqphase=0, nconserved=True):
    #vector corresponding to dipole matrix element with ground state
    retvec=zeros(ntot, dtype='complex')
    if(nconserved):
        nrange=range(1)
    else:
        nrange=range(nmin,nmax+1)
    for i,n in iterproduct(range(nstates), nrange):
        tmpindx=psinindex(i,n,nmin)
        retvec[tmpindx]=dip[0,i]
    retvec=applyfloquetphase(retvec,floqphase)
    return retvec

def adiabaticdipoles(tarray,evecarray,phasearray):
    retarray=zeros(shape(phasearray),dtype='complex')
    for i in range(len(tarray)):
        dipvec=dipg_floq(Hdip,nmin,nmax,0,nconserved=False)
        retarray[i,:]=dot(evecarray[i,:],dipvec)*exp(-1j*phasearray[i])
    return conjugate(retarray)

def plot_adiabatic_spectrum_vs_tion(tarray, ionamparray, adiabaticdiparray,
                                    phasearray, iplt=0, nplt=0, iname=None,**kwargs):
    if(iname!=None):
        iplt=namedict[iname]
    pltindx=psinindex(iplt,nplt,nmin)
    ionvec=ionamparray[:,pltindx]*exp(-1j*phasearray[:,pltindx])
    dipvec=adiabaticdiparray[:,pltindx]
    t1t2diparray=outer(conjugate(ionvec),dipvec)
    indx1array=outer(range(nt),ones(nt))
    indx2array=outer(ones(nt),range(nt))
    fftfreqs=fftdipfreqs(tarray)
    timearray,freqarray=vecstoarrays(tarray,fftfreqs)
    t1t2diparray=where(indx1array<=indx2array,t1t2diparray,0)
    FTt1t2diparray=nDFFT(t1t2diparray,axislist=[1],shiftaxes=True)
    retfig=imshowplot(timearray,freqarray,abs(FTt1t2diparray),**kwargs)
    return retfig

def plot_full_adiabatic_spectrum_vs_tion(tarray, ionamps, adiabaticdiparray,
                                    phasearray, **kwargs):
#    ionvec=ionamps*exp(-1j*phasearray)
#    dipvec=adiabaticdiparray
#    print("shapes\t"+str(shape(conjugate(ionvec)))+"\t"+str(shape(dipvec)))
#    t1t2diparray=dot(conjugate(ionvec),transpose(dipvec))
#    indx1array=outer(range(nt),ones(nt))
#    indx2array=outer(ones(nt),range(nt))
#    t1t2diparray=where(indx1array<=indx2array,t1t2diparray,0)
    t1t2diparray=induceddipoles(tarray,phasearray,ionamparray,adiabaticdiparray)
    FTt1t2diparray=nDFFT(t1t2diparray,axislist=[1],shiftaxes=True)
    fftfreqs=fftdipfreqs(tarray)
    timearray,freqarray=vecstoarrays(tarray,fftfreqs)
    retfig=imshowplot(timearray,freqarray,abs(FTt1t2diparray),**kwargs)
    return retfig

def plot_full_transient_absorption(tarray, ionamps, adiabaticdiparray,
                                   phasearray, **kwargs):
    t1t2diparray=induceddipoles(tarray, phasearray, ionamps, adiabaticdiparray)
    fftfreqs=fftdipfreqs(tarray)
    timearray,freqarray=vecstoarrays(tarray,fftfreqs)
    FTt1t2diparray=nDFFT(t1t2diparray, axislist=[1], shiftaxes=True)
    t0array=outer(tarray,ones(nt))
    t1array=outer(ones(nt),tarray)
    print("shapes\t"+str(shape(conjugate(t0array)))+"\t"+str(shape(t1array)))
    pulsearray=envelope(t1array-t0array,wenv1)*cos(w1*(t1array-t0array))
    ftpulsearray=nDFFT(pulsearray,axislist=[1], shiftaxes=True)
    retfig=imshowplot(timearray,freqarray,2*imag(ftpulsearray*conjugate(FTt1t2diparray)),**kwargs)
    #retfig=imshowplot(t0array*aut/1000,t1array*aut/1000,abs(t1t2diparray))
    return retfig

def plot_transient_absorption_adiabatic_state(tarray, ionamps, adiabaticdiparray,
                                   phasearray,iplot=0, nplot=0, iname=None, **kwargs):
    if(iname!=None):
        iplot=namedict[iname]
    maskvec=zeros(ntot)
    maskvec[psinindex(iplot,nplot,nmin)]=1
    t1t2diparray=induceddipoles(tarray, phasearray, ionamps, adiabaticdiparray, maskvec=maskvec)
    fftfreqs=fftdipfreqs(tarray)
    timearray,freqarray=vecstoarrays(tarray,fftfreqs)
    FTt1t2diparray=nDFFT(t1t2diparray, axislist=[1], shiftaxes=True)
    t0array=outer(tarray,ones(nt))
    t1array=outer(ones(nt),tarray)
    print("shapes\t"+str(shape(conjugate(t0array)))+"\t"+str(shape(t1array)))
    pulsearray=envelope(t1array-t0array,wenv1)*cos(w1*(t1array-t0array))
    ftpulsearray=nDFFT(pulsearray,axislist=[1], shiftaxes=True)
    retfig=imshowplot(timearray,freqarray,2*imag(ftpulsearray*conjugate(FTt1t2diparray)),**kwargs)
    #retfig=imshowplot(t0array*aut/1000,t1array*aut/1000,abs(t1t2diparray))
    return retfig


#def plot_adiabatic_spectrum_vs_tion(tarray, ionamparray, adiabaticdiparray,
#                                    phasearray, iplt=0, nplt=0,  iname=None):
#    if(iname!=None):
#        iplt=namedict[iname]
#    pltindx=psinindex(iplt,nplt,nmin)
#    dipvec=adiabaticdiparray[:,pltindx]
#    n0=len(tarray)
#    diparray=zeros((n0,n0),dtype='complex')
#    for i,j in iterproduct(range(n0),range(n0)):
#        if(j>=i and j<=i+tau0/dt):
#            diparray[i,j]=dipvec[j]*ionamparray[i,pltindx]*exp(-1j*phasearray[i,pltindx])
##    for i in range(n0):
##        diparray[i,i:]=dipvec[i:]*ionamparray[i,pltindx]*exp(-1j*phasearray[i,pltindx])
#    fftdiparray=nDFFT(diparray,axislist=[1], shiftaxes=True)
#    fftfreqs=fftdipfreqs(tarray)
#    xarray,yarray=vecstoarrays(tarray/(1000/aut),fftfreqs*Hrt)
#    Fpulsearray=zeros(shape(diparray),dtype='complex')
#    for i in range(n0):
#        Fpulsearray[i,:]=list(map(lambda t: cos(w1*t)*envelope(t-tarray[i],wenv1),tarray))
#    ftpulsearray=nDFFT(Fpulsearray,axislist=[1],shiftaxes=True)
#    retfig=imshowplot(xarray,yarray,2*imag(conjugate(fftdiparray)*ftpulsearray))
#    return retfig

#@autojit
#def plot_full_adiabatic_spectrum_vs_tion(tarray,ionamparray,adiabaticdiparray):
#    n0=len(tarray)
#    diparray=zeros((n0,n0),dtype='complex')
#    for i,j in iterproduct(range(n0),range(n0)):
#        if(j>=i and j<=i+tau0/dt):
#            diparray[i,j]=sum(ionamparray[i,:]*adiabaticdiparray[j,:]*exp(-1j*phasearray[i,:]))#dipvec[j]*ionamparray[i,pltindx]
#    fftdiparray=nDFFT(diparray,axislist=[1], shiftaxes=True)
#    fftfreqs=fftdipfreqs(tarray)
#    xarray,yarray=vecstoarrays(tarray/(1000/aut),fftfreqs*Hrt)
#    retfig=imshowplot(xarray,yarray,abs(fftdiparray))
#    return retfig

def applyfloquetphase(inpfloqvec, floqphase=0):
#floquet phase is phase proportional to number of photons
    retfloqvec=copy(inpfloqvec)
    for i,n in iterproduct(range(nstates), range(nmin,nmax+1)):
        tmpindx=psinindex(i,n,nmin)
        retfloqvec[tmpindx]*=exp(1j*n*floqphase)
    return retfloqvec

def instEig_floq(H0,dip,Eenv,phaseE,nmin,nmax):
    tmpH=Hfloq(H0,dip,Eenv,phaseE,nmin,nmax)
    evals,evecs=eigh(tmpH)
    srtind=argsort(evals)
    retevals=zeros(shape(evals),dtype='complex')
    retevecs=zeros(shape(evecs),dtype='complex')
    for i in range(len(srtind)):
        retevals[i]=evals[srtind[i]]
        retevecs[i,:]=evecs[:,srtind[i]]
    return retevals,retevecs

def eigenvectormatch(mat1,mat2,envec2,enorder=False):
    #we want to reorder the eigenvectors in mat2 such that mat2[i,:] gives the
    #best overlap with mat1[i,:]
    #To do this, we'll create a list of dot products, sort it, then create a
    #list of tuples such that each successive tuple gives the best match between
    #eigenvectors which have not previously been matched.
    (n0,n1)=shape(mat1)
    nprod=n0*n0
    tmpmat=zeros((n0,n0))
    dotmat=dot(conjugate(mat1),transpose(mat2))
    retmat2=copy(mat2)
    retenvec2=copy(envec2)
    if(not(enorder)):
        #print("not enorder")
        dotmatvec=dotmat.reshape((nprod))
        argdotmatvec=argsort(abs(real(dotmatvec)))[::-1]
        iset=set(range(n0))
        jset=set(range(n0))
        tuplelist=[]
        indx=0
        while(len(iset)>0 and len(jset)>0 and indx<nprod):
            smallindx=argdotmatvec[indx]
            indx+=1
            smalli=int(floor(smallindx/n0))
            smallj=mod(smallindx,n0)
            if((smalli in iset) and (smallj in jset)):
                iset.remove(smalli)
                jset.remove(smallj)
                tuplelist.append((smalli, smallj)) 
#    print("indx "+str(indx))
#    print("tuplelist "+str(tuplelist))
        for k in range(len(tuplelist)):
            (i,j)=tuplelist[k]
            retmat2[i,:]=mat2[j,:]
            retenvec2[i]=envec2[j]
            if(real(dotmat[i,j])<0):
                retmat2[i,:]*=-1
    if(enorder):
        print("enorder")
        for i in range(n0):
            if(real(dotmat[i,i]<0)):
                retmat2[i,:]*=-1
    return retenvec2, retmat2
    

def adjustevecs(evalarray,evecarray,enorder=False):
    (nt,ns)=shape(evalarray)
    retevalarray=copy(evalarray)
    retevecarray=copy(evecarray)
    diagmat=diag(ones(ntot))#start with same ordering as psinindex
    retevalarray[0],retevecarray[0,:,:]=eigenvectormatch(diagmat,retevecarray[0,:,:],
                                                         retevalarray[0,:],
                                                         enorder=enorder)
    for i in range(1,nt):
        retevalarray[i], retevecarray[i,:,:] = eigenvectormatch(retevecarray[i-1,:, :],
                                                           retevecarray[i, :, :],
                                                           retevalarray[i, :],
                                                           enorder=enorder)
    return retevalarray, retevecarray

def groundstatedip(evecarray,t=0,wphase=False, sign=1):
    dipvec=zeros(ntot, dtype='complex')
    for i,n in iterproduct(range(nstates), range(nmin,nmax+1)):
        tmpindx=psinindex(i,n,nmin)
        dipvec[tmpindx]=Hdip[0,i]
        if(wphase):
            dipvec[tmpindx]*=exp(1j*sign*w0*n*t)
    return dot(evecarray,dipvec)

@autojit
def accumulated_phases(tarray,evalarray):
    phasearray=zeros(shape(evalarray),dtype='complex')
    for i in range(1,len(tarray)):
        dt=tarray[i]-tarray[i-1]
        phasearray[i,:]=phasearray[i-1,:]-dt*(evalarray[i-1,:]+evalarray[i,:])/2
    return phasearray

def adiabatic(tarray,evalarray,evecarray,wphase=True):
    phasearray=accumulated_phases(tarray,evalarray)
    (n0,n1,n2)=shape(evecarray)
    diparray=zeros((n0,n1),dtype='complex')
    for i in range(len(tarray)):
        t=tarray[i]
        diparray[i,:]=groundstatedip(evecarray[i,:,:],t,wphase,sign=-1)
    retarray=exp(1j*phasearray)*diparray
    return retarray

    

################################################################################

def instEig(t,Ef):
    Ef=Efield(t)
    evals,evecs=eig(H0+Hd*Ef)
    srtind=argsort(evals)
    retevals=zeros(shape(evals))
    retevecs=zeros(shape(evecs))
    for i in range(len(srtind)):
        retevals[i]=evals[srtind[i]]
        retevecs[:,i]=evecs[:,srtind[i]]

    return retevals,retevecs

def adjustevecarray(inpevecarray):
    retarray=copy(inpevecarray)
    (nt,n0,n1)=shape(retarray)
    for n in range(n1):
        indx=abs(retarray[0,:,n]).argmax()
        if(retarray[0,indx,n]<0):
            retarray[0,:,n]*=-1
    for i in range(1,nt):
        for n in range(n1):
            dp=dot(retarray[i,:,n],retarray[i-1,:,n])
            if(dp<0):
                retarray[i,:,n]*=-1
    return retarray

def phaseaccumulation(evalarray,dt):
    (nt,nv)=shape(evalarray)
    retarray=zeros(shape(evalarray), dtype='complex')
    for i in range(1,nt):
        retarray[i,:]=retarray[i-1,:]+(evalarray[i,:]+evalarray[i-1,:])/2*dt
    return retarray

def nDFFT(cinparray,axislist,shiftaxes=False, inv=False):
    if(inv and shiftaxes):
        tmpcinparray=ifftshift(cinparray,axes=axislist)
        fftdiparray=ifftn(tmpcinparray,axes=axislist)
        fftdiparray=ifftshift(fftdiparray,axes=axislist)
    if(not(inv) and shiftaxes):
        fftdiparray=fftn(cinparray,axes=axislist)
        fftdiparray=fftshift(fftdiparray,axes=axislist)
    if(not(inv) and not(shiftaxes)):
        fftdiparray=fftn(cinparray,axes=axislist)
    if(inv and not(shiftaxes)):
        fftdiparray=ifftn(cinparray,axes=axislist)
    return fftdiparray

def fftdipfreqs(tvec):
    ntvec=len(tvec)
    dt=tvec[1]-tvec[0]
    retvec=fftfreq(ntvec,dt)*2*pi
    retvec.sort()
    return retvec

def nonadiabatictransitions(inpevecarray,dt):
    retarray=zeros(shape(inpevecarray))
    (nt,n0,n1)=shape(inpevecarray)
    for i in range(1,nt-1):
        diff=(inpevecarray[i+1,:,:]-inpevecarray[i-1,:,:])/dt
        retarray[i,:,:]=dot(transpose(inpevecarray[i,:,:]),diff)
    return retarray

@autojit
def twoDdip_adiabatic(amparray,diparray,phasearray):
    (nt,ns)=shape(amparray)
#first, find dipole at time t2 due to ionization at time t1
    tmparray=zeros((nt,nt),dtype='complex')

    print("shape amparray\t"+str(shape(amparray)))
    print("shape phasearray\t"+str(shape(phasearray)))
    print("ntstart\t"+str(ntstart))
    #tmpamp=amparray*exp(-1j*phasearray)
    tmpamp=zeros((ntstart,ns),dtype='complex')
    for i,j in product(range(ntstart),range(ns)):
        tmpamp[i,j]=amparray[i,j]*exp(-1j*phasearray[i,j])

    tmpret=diparray*exp(1j*phasearray)
    tmparray=dot(tmpamp,transpose(tmpret))
#zeros for tf<ti
    for j in range(ntstart):
        for i in range(j):
            tmparray[j,i]=0.

    return tmparray

def twoDdip_populate_eigenstates(amparray,evecarray,Efieldfree,dipfieldfree):
#populate field free eigenstates according to field-on eigenstates
    (nt,ns)=shape(amparray)
    excitationamps=zeros((nt,ns),dtype='complex')
    retarray=zeros((nt,nt),dtype='complex')
    for i in range(nt):
        excitationamps[i,:]=dot(evecarray[i,:,:],amparray[i,:])*exp(-1j*Efieldfree*tarray[i])

    tmpdip=zeros((nt,ns),dtype='complex')
    for i in range(nt):
        tmpdip[i,:]=dipfieldfree[:]*exp(1j*tarray[i]*Efieldfree[:])

    retarray=dot(excitationamps,transpose(tmpdip))
#zeros for tf<ti
    for j in range(nt):
        for i in range(j):
            retarray[j,i]=0.
    return retarray
    

#    for i in range(nt):
#        tmpamp=amparray[i,:]*exp(1j*phasearray[i,:])
#        for j in range(i,nt):
#            tmpdip=exp(-1j*phasearray[j,:])*diparray[j,:]
#            retarray[i,j]=dot(tmpamp,tmpdip)
    return retarray

################################################################################
#plotting routines
#def plotevecn(tarray,evecarray,i=0,n=0, tol=.01, iname=None):
#    if(iname!=None):
#        i=namedict[iname]
#    tmpindx0=psinindex(i,n,nmin)
#    tmpfig=plt.figure()
#    tmpfig.suptitle("adiabatic eigenvector i="+namelist[i]+", n= "+str(n))
#    for tmpi,tmpn in iterproduct(range(nstates), range(nmin,nmax+1)):
#        tmpindx1=psinindex(tmpi,tmpn,nmin)
#        if(max(abs(evecarray[:,tmpindx0,tmpindx1]))>=tol):
#            plt.plot(tarray,evecarray[:,tmpindx0,tmpindx1], label="i= "+namelist[tmpi]+", n= "+str(tmpn), color=styles[tmpindx1][0], linestyle=styles[tmpindx1][1], linewidth=2)
#    plt.legend()
#    return tmpfig

def instr(i,n):
    return "i="+namelist[i]+", n="+str(n)

def plotevecn(tarray,evecarray,i=0,n=0,tol=.01,iname=None, linewidth=2):
    if(iname!=None):
        i=namedict[iname]
    tmpindx0=psinindex(i,n,nmin)
    title="adiabatic eigenvector "+instr(i,n)
    return plotarray(tarray,real(evecarray[:,tmpindx0,:]),tol=tol,title=title,linewidth=linewidth)

    

def plotadiabaticspectrum(i=0,n=0,xlo=0,xhi=30, iname=None):
    if(iname!=None):
        i=namedict[iname]
    fignext=plt.figure()
    tmpindx=psinindex(i,n,nmin)
    fignext.suptitle("spectrum due to adiabatic state i="+namelist[i]+", n= "+str(n))
    plt.plot(fftfreqs*Hrt,abs(fftadiabaticdips[:,tmpindx]),
             label="i= "+namelist[i]+", n= "+str(n), linewidth=2)
    plt.xlim((xlo,xhi))
    return fignext

def plotarray(tarray,pltarray, tol=.01, title="", linewidth=2):
    retfig=plt.figure()
    retfig.suptitle(title)
    (n0,n1)=shape(pltarray)
    maxvalvec=list(map(lambda i:max(abs(pltarray[:,i])), range(n1)))
    maxval=max(maxvalvec)
    #print("maxvalvec"+str(maxvalvec))
    srtindices=argsort(maxvalvec)[::-1]
    nline=0
    #print("starting value "+str(maxvalvec[srtindices[nline]]))
    while(maxvalvec[srtindices[nline]]>tol*maxval and nline<(n1-1)):
        nplot=srtindices[nline]
        #print("nplot "+str(nplot))
        itmp, ntmp=psinindexinv(nplot,nmin)
        plt.plot(tarray,pltarray[:,nplot], label=instr(itmp,ntmp),
                 color=styles[nline][0], linestyle=styles[nline][1], linewidth=linewidth)
        nline+=1
    plt.legend(loc='best')
    return retfig



################################################################################

#First, set up Hamiltonian for He excited states
nstates=6#4#5
Eg=0.+0j
E1s2s=20.616
E1s2p=21.22
E1s3s=22.92
E1s3p=23.08
E1s3d=23.09
Efieldfree=array([Eg,E1s2s,E1s2p,E1s3s,E1s3p,E1s3d],dtype='complex')/Hrt
#Efieldfree=array([Eg,E1s2s,E1s2p,E1s3s,E1s3p])/Hrt
H0=diag(Efieldfree)

namelist=['$1s^{2}$','1s2s','1s2p','1s3s','1s3p','1s3d']
namedict={'$1s^{2}$':0,'1s2s':1,'1s2p':2,'1s3s':3,'1s3p':4,'1s3d':5,'1ssq':0}

Hdip=zeros((nstates,nstates),dtype='complex')
#Hdipole matrix elements with ground state
Hdip[0,2]=.41998639042511915
Hdip[2,0]=Hdip[0,2]
Hdip[0,4]=0.20761245395206396
Hdip[4,0]=Hdip[0,4]

#Hdipole matrix elements between excited states
Hdip[1,2]=2.909870044539534
Hdip[2,1]=Hdip[1,2]
Hdip[1,4]=0.91063937751587065
Hdip[4,1]=Hdip[1,4]

Hdip[3,2]=1.8631364216534694
Hdip[2,3]=Hdip[3,2]
Hdip[3,4]=7.1331830032922072
Hdip[4,3]=Hdip[3,4]

Hdip[5,2]=3.0535203080771018
Hdip[2,5]=Hdip[5,2]
Hdip[5,4]=8.1241978011255487
Hdip[4,5]=Hdip[5,4]

#Hdip*=-1
#Hdip/=(2*pi)

################################################################################
#pulse parameters
w0=.0565
w1=25/Hrt
F0=sqrt(3e12/Iat)#.01
F1=sqrt(1e10/Iat)
ncycles=8

tau0=11e3/aut
tau1=330/aut#330/aut
wenv0=(pi/2)/tau0
wenv1=(pi/2)/tau1
t0=-2*tau0
tf=2*tau0
dt=2#.1
tarray=arange(t0,tf,dt)
ntstart=2*int(abs(t0/dt))
nt=len(tarray)
evlo=15
evhi=40

nmin=-8
nmax=8
numn=nmax-nmin+1
ntot=nstates*numn
ncolors=int(ceil(ntot/4))
dcolorphi=1.25
colorcycle=list(map(plt.cm.hsv,mod(array(range(ncolors))*dcolorphi/(2*pi),1)))
linestyles=['-','--','-.',':']
#styles=[(color,linestyle) for linestyle in linestyles for color in colorcycle] 
#styles=list(iterproduct(linestyles,colorcycle))#list(iterproduct(colorcycle,linestyles))
styles=[]
for ls in linestyles:
    for c in colorcycle:
        styles.append([c,ls])

enarray=zeros((nt,ntot),dtype='complex')
evecarray=zeros((nt,ntot,ntot),dtype='complex')
sCEP=0#streaking CEP phase
for i in range(nt):
    enarray[i, :],  evecarray[i, :, :]=instEig_floq(H0, Hdip,
                                                    F0*envelope(tarray[i], wenv0),
                                                    sCEP, nmin, nmax)

#adjust evecs to eliminate factors of -1 due to arbitrary sign convention in eig routine
#evecarray=adjustevecarray(evecarray)
enarray, evecarray=adjustevecs(enarray,evecarray,enorder=False)

fig0=plt.figure()
fig0.suptitle("energies")
for i,n in iterproduct(range(nstates), range(nmin,nmax+1)):
    tmpindx=psinindex(i,n,nmin)
    plt.plot(Hrt*enarray[:,tmpindx],label="i= "+namelist[i]+", n= "+str(n),
             color=styles[tmpindx][0],linestyle=styles[tmpindx][1], linewidth=2)
plt.legend()

#plot eigenvectors as a function of time
fign=plotevecn(tarray,evecarray,iname="1s3d",n=-1)

#accumulated phase
phasearray=phaseaccumulation(enarray,dt)
print("shape phasearray\t"+str(shape(phasearray)))
#ionization amplitudes
pulseCEP=0
ionamparray=ionamparray(F1,pulseCEP,enarray,phasearray,evecarray,tarray,w1,wenv1,nglpts=5)
#projvec=zeros(ntot, dtype=complex)
#projvec[psinindex(namedict["1s3p"],0,nmin)]=1
#ionamparray=ionamparray_fieldfreeprojection(evecarray,projvec)

#ionamparray=ionamparray_test(evecarray,nffstate=0,iffname='1s3p')

ffionamparray=fieldfree_ionamparray(ionamparray,evecarray)

fignext=plotarray(tarray,abs(ionamparray),title="adiabatic eigenstate ionization amplitudes", tol=.1)
fignext=plotarray(tarray,abs(ffionamparray),title="field free ionization amplitudes", tol=.1)



#dipoles with ground state
adiabaticdips=adiabaticdipoles(tarray,evecarray,phasearray)
fftadiabaticdips=nDFFT(adiabaticdips,axislist=[0],shiftaxes=True)
fftfreqs=fftdipfreqs(tarray)
plotadiabaticspectrum(1,1,xlo=18,xhi=35)

plt.show()

