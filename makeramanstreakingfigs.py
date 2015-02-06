import sys
sys.path.append(r'/Users/zwalters/pythonscripts/')
from twoDplot import *
from numpy.fft import *
################################################################################
#load pickled files

Hrt=27.21
aut=24.2
Iat=3.51e16#atomic unit of intensity

wvec=load("wvec.pkl")
tmeasurevec=load("tmeasurevec.pkl")
nupvec=load("nupvec.pkl")
ndownvec=load("ndownvec.pkl")
nstreakvec=load("nstreakvec.pkl")
n_vs_tarray=load("n_vs_tarray.pkl")
n_vs_warray=load("n_vs_warray.pkl")
dtdownvec=load("dtdownvec.pkl")
dtstreakvec=load("dtstreakvec.pkl")
CEPupvec=load("CEPupvec.pkl")
CEPdownvec=load("CEPdownvec.pkl")
CEPstreakvec=load("CEPstreakvec.pkl")



################################################################################
#function definitions
def nm_to_eV(nm):
    hc=1239.841#planck's constant in eV*nm
    eV=hc/nm
    return eV
w0=nm_to_eV(800)/Hrt

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

def removestreakingphase(nt0=0, nt1=0, nup=1,ndown=-1, wstreak=w0):
    upindx=list(nupvec).index(nup)
    downindx=list(ndownvec).index(ndown)
    nvstarray=copy(n_vs_tarray[nt0, nt1, upindx, downindx, :, :])
    for i in range(len(nstreakvec)):
        nvstarray[i, :]*=exp(-1j*nstreakvec[i]*tmeasurevec*wstreak)
    return nvstarray

def removestreakingphase_2DFFT(nt0=0, nt1=0, nup=1,ndown=-1, wstreak=w0):
    nvstarray=removestreakingphase(nt0, nt1, nup, ndown, wstreak)
    nvswarray=nDFFT(nvstarray,axislist=[1], shiftaxes=True)
    phivswarray=nDFFT(nvswarray,axislist=[0], shiftaxes=True, inv=True)
    return phivswarray
    

################################################################################
#main program
 

#Plot nup=1, ndown=-1 
upindx=list(nupvec).index(1)
downindx=list(ndownvec).index(-1)
xarray,yarray=vecstoarrays(nstreakvec,wvec)
fig0=contourplot(xarray,yarray,n_vs_warray[0,0,upindx,downindx,:,:],
                    ymultfactor=Hrt, ncontours=18)

phi_vs_wparray=removestreakingphase_2DFFT()
xarray,yarray=vecstoarrays(CEPstreakvec, wvec)
fig1=imshowplot_hsv(xarray,yarray, phi_vs_wparray, ymultfactor=Hrt, colorpower=1, logrange=log(1e2))

fig2=contourplot(xarray,yarray, phi_vs_wparray, ymultfactor=Hrt, ncontours=8)
plt.show()
