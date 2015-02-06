from numpy import *
from numpy.linalg import eig
from numpy.fft import *
from numpy.polynomial.legendre import leggauss
from itertools import product
import matplotlib.pyplot as plt
from numba import jit, autojit
from itertools import product as iterproduct
import sys
sys.path.append(r'/Users/zwalters/pythonscripts')
from twoDplot import *
Hrt=27.21
aut=24.2
Iat=3.51e16#atomic unit of intensity

def envelope(t,wenv):
    if(abs(wenv*t)<pi/2):
        retval=pow(cos(wenv*t),2)
    else:
        retval=0
    return retval

def pulse(t,t0,w,wenv):
    return cos(w*(t-t0))*envelope(t-t0,wenv)

def Efield(t):
    return E0*envelope(t,wenv0)*cos(w0*t)

def twopulses(t,E0,w0,tau0,tc0,E1,w1,tau1,tc1):
    wenv0=(pi/2)/tau0
    wenv1=(pi/2)/tau1
    retval= (E0*envelope((t-tc0),wenv0)*cos(w0*(t-tc0)) +
             E1*envelope((t-tc1),wenv1)*cos(w1*(t-tc1)))
    return retval

def goldenrule(wosc,En,tau):
    dE=wosc-En
    return (2/dE)*sin(dE*tau)

def goldenruleamps(diparray,enarray,wosc,tau):
    retarray=zeros(shape(enarray))
    (nt,ns)=shape(enarray)
    for i,j in product(range(nt), range(ns)):
        retarray[i,j]=diparray[i,j]*goldenrule(wosc,enarray[i,j],tau)
    return retarray

nglpts=20
pts,wts=leggauss(nglpts)
def rotwave(tau,wosc,Eoft,phiCEP):
    tpts=pts*tau*(pi/2)
    twts=pts*tau*(pi/2)
    vals=list(map(lambda t: exp(1j*((wosc-Eoft)*t+phiCEP))*pow(cos(t/tau),2),tpts))
    return sum(vals*twts)

def rotwaveamps(diparray,enarray,wosc,tau,phiCEP=0.):
    retarray=zeros(shape(enarray),dtype='complex')
    (nt,ns)=shape(enarray)
    for i,j in product(range(nt),range(ns)):
        retarray[i,j]=rotwave(tau,wosc,enarray[i,j],phiCEP)*exp(1j*w1*tarray[i])
    return retarray



def instEig(t,Ef):
    Ef=Efield(t)
    evals,evecs=eig(H0+Hd*Ef)
    srtind=argsort(evals)
    retevals=zeros(shape(evals), dtype='complex')
    retevecs=zeros(shape(evecs), dtype='complex')
    for i in range(len(srtind)):
        retevals[i]=evals[srtind[i]]
        retevecs[i,:]=evecs[:,srtind[i]]

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
    retarray=zeros(shape(evalarray))
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
    retarray=zeros(shape(inpevecarray), dtype='complex')
    (nt,n0,n1)=shape(inpevecarray)
    for i in range(1,nt-1):
        diff=(inpevecarray[i+1,:,:]-inpevecarray[i-1,:,:])/dt
        mean=(inpevecarray[i+1,:,:]+inpevecarray[i-1,:,:])/2
        retarray[i,:,:]=dot(conjugate(diff),transpose(mean))
    return retarray

def phaseaccumulation(evalarray,dt):
    (nt,nv)=shape(evalarray)
    retarray=zeros(shape(evalarray), dtype='complex')
    for i in range(1,nt):
        retarray[i,:]=retarray[i-1,:]+(evalarray[i,:]+evalarray[i-1,:])/2*dt
    return retarray

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

def adiabaticdipoles(phasearray, evecarray,dipvec):
    return conjugate(exp(-1j*phasearray)*dot(evecarray,dipvec))

def twoDspectra_adiabatic(ionamparray, phasearray, evecarray, dipvec):
    adiabaticdiparray=adiabaticdipoles(phasearray,evecarray,dipvec)
    (nt,ns)=shape(adiabaticdiparray)
    retarray=zeros((nt,nt,ns), dtype='complex')
    tomat=timeordermat(nt)
    for i in range(ns):
        retarray[:,:,i]=outer(exp(-1j*phasearray[:,i])*ionamparray[:,i],
                              adiabaticdiparray[:,i])*tomat
    return retarray


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


def envelope(t,wenv):
    return where(abs(wenv*t)<pi/2,pow(cos(wenv*t),2),0)

#def ionamps(F0,pulseCEP,Envec,phasevec,evecarray,tion,w,wenv,nglpts=5):
def ionamps_adiabatic(F0,w, wenv, tion, envec, dipvec, nglpts=5):
    inttarray, intwtarray=ionintpts(tion,w,wenv,nglpts)
    pulsevec=list(map(lambda t: cos((t-tion)*w)*envelope(t-tion,wenv), inttarray))
    enoscarray=array(list(map(lambda t: exp(1j*(t-tion)*envec), inttarray)))
    retarray=zeros(nstates, dtype='complex')
    for i in range(nstates):
        retarray[i]=sum(pulsevec*enoscarray[:,i]*intwtarray)
    retarray*=1j*F0*dipvec
    return retarray

def fieldfreeprojection_ionamparray(evecarray,projvec):
    return dot(evecarray,projvec)

def adiabatic_ionamparray(F0, w, wenv, tarray, enarray, diparray, nglpts=5):
    retarray=zeros((nt,nstates), dtype='complex')
    for i in range(len(tarray)):
        retarray[i,:]=ionamps_adiabatic(F0,w,wenv,tarray[i], enarray[i],
                                        diparray[i,:], nglpts)
    return retarray

def twoDpulsearray(tarray):
    retarray=zeros((nt,nt),dtype='complex')
    pulse0vec=E0*array(list(map(lambda t: pulse(t,0,w0,wenv0), tarray)))
#    for i in range(nt):
#        t0=tarray[i]
#        pulse1vec=E1*array(list(map(lambda t: pulse(t,t0,w1,wenv1), tarray)))
#        retarray[i,:]=pulse1vec+pulse0vec
    dtmat=deltatmat(tarray)
    t0mat=outer(ones(nt),tarray)
    t1mat=outer(tarray,ones(nt))
    pulse0array=pulse(t0mat,0,w0,wenv0)
    pulse1array=pulse(t0mat,t1mat,w1,wenv1)#pulse(dtmat,0,w1,wenv1)#array(map(lambda t: pulse(t,0,w1,wenv1),dtmat))
    #toarray=timeordermat(nt)
    retarray=pulse0array+pulse1array
    return retarray
    

#@autojit
#def twoDdip_adiabatic(amparray,diparray,phasearray):
#    (nt,ns)=shape(amparray)
##first, find dipole at time t2 due to ionization at time t1
#    tmparray=zeros((nt,nt),dtype='complex')
#
#    print("shape amparray\t"+str(shape(amparray)))
#    print("shape phasearray\t"+str(shape(phasearray)))
#    print("ntstart\t"+str(ntstart))
#    #tmpamp=amparray*exp(-1j*phasearray)
#    tmpamp=zeros((ntstart,ns),dtype='complex')
#    for i,j in product(range(ntstart),range(ns)):
#        tmpamp[i,j]=amparray[i,j]#*exp(1j*phasearray[i,j])
#
#    tmpret=diparray*exp(-1j*phasearray)
#    tmparray=dot(tmpamp,transpose(tmpret))
##zeros for tf<ti
#    for j in range(ntstart):
#        for i in range(j):
#            tmparray[j,i]=0.
#
#    return tmparray
#
##def twoDdip_populate_eigenstates(amparray,evecarray,Efieldfree,dipfieldfree):
###populate field free eigenstates according to field-on eigenstates
##    (nt,ns)=shape(amparray)
##    excitationamps=zeros((nt,ns),dtype='complex')
##    retarray=zeros((nt,nt),dtype='complex')
##    for i in range(nt):
##        excitationamps[i,:]=dot(evecarray[i,:,:],amparray[i,:])*exp(-1j*Efieldfree*tarray[i])
##
##    tmpdip=zeros((nt,ns),dtype='complex')
##    for i in range(nt):
##        tmpdip[i,:]=dipfieldfree[:]*exp(1j*tarray[i]*Efieldfree[:])
##
##    retarray=dot(excitationamps,transpose(tmpdip))
###zeros for tf<ti
##    for j in range(nt):
##        for i in range(j):
##            retarray[j,i]=0.
##    return retarray
#    
#
##    for i in range(nt):
##        tmpamp=amparray[i,:]*exp(1j*phasearray[i,:])
##        for j in range(i,nt):
##            tmpdip=exp(-1j*phasearray[j,:])*diparray[j,:]
##            retarray[i,j]=dot(tmpamp,tmpdip)
#    return retarray
#
################################################################################
#match eigenvectors

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
    diagmat=diag(ones(ns))#start with same ordering as psinindex
    retevalarray[0],retevecarray[0,:,:]=eigenvectormatch(diagmat,retevecarray[0,:,:],
                                                         retevalarray[0,:],
                                                         enorder=enorder)
    for i in range(1,nt):
        retevalarray[i], retevecarray[i,:,:] = eigenvectormatch(retevecarray[i-1,:, :],
                                                           retevecarray[i, :, :],
                                                           retevalarray[i, :],
                                                           enorder=enorder)
    return retevalarray, retevecarray



################################################################################
#First, set up Hamiltonian for He excited states
nstates=6#4#5
Eg=0.
E1s2s=20.616
E1s2p=21.22
E1s3s=22.92
E1s3p=23.08
E1s3d=23.09
Efieldfree=array([Eg,E1s2s,E1s2p,E1s3s,E1s3p,E1s3d])/Hrt
H0=diag(Efieldfree)
#H0=diag([E1s2s,E1s2p,E1s3s,E1s3p])/Hrt

namelist=['$1s^{2}$','1s2s','1s2p','1s3s','1s3p','1s3d']
namedict={'$1s^{2}$':0,'1s2s':1,'1s2p':2,'1s3s':3,'1s3p':4,'1s3d':5,'1ssq':0}

Hd=zeros((nstates,nstates),dtype='complex')
#Hdole matrix elements with ground state
Hd[0,2]=.41998639042511915
Hd[2,0]=Hd[0,2]
Hd[0,4]=0.20761245395206396
Hd[4,0]=Hd[0,4]

#Hdole matrix elements between excited states
Hd[1,2]=2.909870044539534
Hd[2,1]=Hd[1,2]
Hd[1,4]=0.91063937751587065
Hd[4,1]=Hd[1,4]

Hd[3,2]=1.8631364216534694
Hd[2,3]=Hd[3,2]
Hd[3,4]=7.1331830032922072
Hd[4,3]=Hd[3,4]

Hd[5,2]=3.0535203080771018
Hd[2,5]=Hd[5,2]
Hd[5,4]=8.1241978011255487
Hd[4,5]=Hd[5,4]

dipvec=Hd[0,:]



#pulse parameters
w0=.0565
w1=22/Hrt
E0=sqrt(3e12/Iat)#.01
E1=sqrt(1e10/Iat)
ncycles=8

tau0=11e3/aut
tau1=330/aut
wenv0=(pi/2)/tau0
wenv1=(pi/2)/tau1
t0=-1*tau0
tf=5*tau0
dt=2#.1
tarray=arange(t0,tf,dt)
ntstart=2*int(abs(t0/dt))
nt=len(tarray)
evlo=15
evhi=40

enarray=zeros((nt,nstates), dtype='complex')
diparray=zeros((nt,nstates), dtype='complex')
evecarray=zeros((nt,nstates,nstates), dtype='complex')

#as a function of the time within the pulse, find instantaneous eigenvalues &
#eigenvectors
for i in range(nt):
    enarray[i,:], evecarray[i,:,:]=instEig(tarray[i],E0)

#adjust evecs to eliminate factors of -1 due to arbitrary sign convention in eig routine
#evecarray=adjustevecarray(evecarray)
enarray,evecarray=adjustevecs(enarray,evecarray)
fig0=plt.figure()
plt.title("energies vs time")
for i in range(nstates):
    plt.plot(tarray*aut/1000,enarray[:,i]*Hrt, label=namelist[i])
plt.legend()

phasearray=phaseaccumulation(enarray,dt)

#calculate instantaneous dipole moment between eigenstates and ground state
diparray=adiabaticdipoles(phasearray, evecarray, dipvec)#E1*dot(evecarray,dipvec)
fftdiparray=nDFFT(diparray,axislist=[0],shiftaxes=True)
freqarray=fftdipfreqs(tarray)
#fig1=plt.figure()
#plt.title("dipoles vs time")
#for i in range(nstates):
#    plt.plot(tarray*aut/1000, real(diparray[:,i]), label=str(i))
#plt.legend()

fig1p=plt.figure()
plt.title("dipole frequencies")
for i in range(nstates):
    plt.plot(freqarray*Hrt,abs(fftdiparray[:,i]), label=namelist[i])
plt.legend()

##ionization amplitudes
#ionamparray=adiabatic_ionamparray(E1, w1, wenv1, tarray, enarray, diparray,
#                                  nglpts=5)
##test
#ionamparray[:,namedict["1s2s"]]*=20

projvec=zeros(nstates,dtype='complex')
projvec[namedict["1s3p"]]=1
ionamparray=fieldfreeprojection_ionamparray(evecarray,projvec)

fig2=plt.figure()
plt.title("ionization amplitudes")
for i in range(nstates):
    plt.plot(tarray, abs(ionamparray[:,i]), label=namelist[i])
plt.legend()

##Nonadiabatic transitions
#nonadiabaticmat=nonadiabatictransitions(evecarray,dt)
#for i in range(nstates):
#    fignext=plt.figure()
#    plt.title("nonadiabatic transitions, state "+str(i))
#    for j in range(nstates):
#        plt.plot(tarray*aut/1000, nonadiabaticmat[:,i,j])

twoDspectrumarray=twoDspectra_adiabatic(ionamparray, phasearray, evecarray,
                                        dipvec)
fttwoDspectrumarray=nDFFT(twoDspectrumarray,axislist=[1],shiftaxes=True)
freqarray=fftdipfreqs(tarray)
plottarray,plotfarray=vecstoarrays(tarray,freqarray)
for i in range(1,nstates):
    fignext=imshowplot(plottarray*aut/1000, plotfarray*Hrt,
                       abs(fttwoDspectrumarray[:,:,i]), ylo=20, yhi=30,
                       legend=namelist[i])

sumtwoDspectrumarray=sum(twoDspectrumarray,axis=2)
ftsumtwoDspectrumarray=nDFFT(sumtwoDspectrumarray,axislist=[1],shiftaxes=True)
twodpulse=twoDpulsearray(tarray)
fttwodpulse=nDFFT(twodpulse,axislist=[1],shiftaxes=True)
fignext=imshowplot(plottarray*aut/1000, plotfarray*Hrt,
                   real(conjugate(ftsumtwoDspectrumarray)*fttwodpulse), ylo=20, yhi=30,
                   legend="sum")


plt.show()
