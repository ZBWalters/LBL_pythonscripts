import glob
import argparse
from numpy import *
import numpy as np
from numpy.linalg import solve
from itertools import *
import subprocess
import shutil
import os.path
from numba import jit, autojit

#def uniquevals(lst):
#    #returns unique elements in a list while preserving order
#    checked=[]
#    for elt in lst:
#        if elt not in checked:
#            checked.append(elt)
#    return checked

def uniquevals(lst,idfun=None):
    #returns elts in a list with unique id function values (unique
    #elts is idfun=None), while preserving the order of the original
    #list
    if(idfun is None):
        def idfun(x): return x
    seen={}
    result=[]
    for elt in lst:
        marker=idfun(elt)
        if marker in seen: continue
        seen[marker]=1
        result.append(elt)
    return result

def arrayfromfile(filename,rvals=False):
    inparray=genfromtxt(filename,comments="#")
    #print "shape inparray "+ str(shape(inparray))
    (n,m)=shape(inparray)
    ncols=int((m-1)/2)
    retarray=[]
    if(rvals):#return only the real components
        retarray.append(inparray[:,0]+0j)
        for i in range(ncols):
            retarray.append(inparray[:,2*i+1]+0j)
    else:#return the real and imaginary components
        retarray.append(inparray[:,0])
        for i in range(ncols):
            retarray.append(inparray[:,2*i+1]+1j*inparray[:,2*i+2])
    return array(retarray)

def readarray(readindx,filenamestr,rvals=False):
    filename=str(readindx)+filenamestr#+"/ZDipoleexpect.Dat"
    return arrayfromfile(filename,rvals)


def makediparray(dipcol,ntxuv,nCEPxuv,nCEPir,arraylist):
    (ncols,nt)=shape(arraylist[0])
    tvec=arraylist[0][0,:]
    retarray=zeros((ntxuv,nCEPxuv,nCEPir,nt))*0j
    for (itxuv,iCEPxuv,iCEPir) in \
        product(range(ntxuv),range(nCEPxuv),range(nCEPir)):
        indx=itxuv*(nCEPxuv*nCEPir)+iCEPxuv*(nCEPir)+iCEPir
        #print("indx\t"+str((itxuv,iCEPxuv,iCEPir))+"\t"+str(indx))
        retarray[itxuv,iCEPxuv,iCEPir,:]=arraylist[indx][dipcol,:]
    return tvec,retarray

def makediparray_raman(dipcol,arrayshape,arraylist):
    (ndt1,ndt2,nCEP1,nCEP2,nCEP3)=arrayshape
    tvec=arraylist[0][0,:]
    ntpts=len(tvec)
    retarray=zeros((ndt1,ndt2,nCEP1,nCEP2,nCEP3,ntpts),dtype=cfloat)
    readindx=0
    for i,j,k,l,m in product(range(ndt1), range(ndt2), range(nCEP1),
                             range(nCEP2), range(nCEP3)):
        retarray[i,j,k,l,m,:]=arraylist[readindx][dipcol,:]
        readindx+=1
    return tvec, retarray

def kIRtransformdiparray(diparray):
    retarray=fft.fft(diparray,axis=-2)
    retarray=fft.fftshift(retarray,axes=(-2))
    return retarray

def FFTdiparray(diparray):
    retarray=fft.fft(diparray,axis=-1)
    retarray=fft.fftshift(retarray,axes=(-1))
    return retarray


def readparameterkey_raman(inpfilename):
    #read parameters for the calculation from the file written by the
    #setup script
    inparray=genfromtxt(inpfilename,comments="#")
    print("shape inparray\t"+str(shape(inparray)))
    dtdownvec=sort(list(uniquevals(inparray[:,1])))
    dtstreakvec=sort(list(uniquevals(inparray[:,2])))
    CEPupvec=sort(list(uniquevals(inparray[:,3])))
    CEPdownvec=sort(list(uniquevals(inparray[:,4])))
    CEPstreakvec=sort(list(uniquevals(inparray[:,5])))
    return dtdownvec,dtstreakvec,CEPupvec,CEPdownvec,CEPstreakvec

@autojit
def Linsys(CEParray,Rvecarray):
    Lmat=zeros((len(CEParray),len(Rvecarray)))*0j
    for i in range(len(CEParray)):
        for j in range(len(CEParray)):
            Lmat[i,j]=exp(1j*pi*dot(CEParray[i],Rvecarray[j]))
    return Lmat

def swapindx(i,i1,i2):
    retval=i
    if(i==i1):
        retval= i2
    if(i==i2):
        retval=i1
    return retval

@autojit
def swaptuple(inptuple,i1,i2):
    #for a given input tuple, interchange all instances of i1 or i2
    #with i2 or i1, respectively
    retlist=list(inptuple)
    retlist[i1], retlist[i2]=retlist[i2], retlist[i1]
    rettuple=tuple(retlist)
    return rettuple

@autojit
def swaparray(inparray,i1,i2):
    inpshape=shape(inparray)
    (n1,n2,n3,n4)=inpshape
    retshape=swaptuple(inpshape,i1,i2)
    #print("inpshape, retshape\t"+str(inpshape)+"\t"+str(retshape))
    retarray=zeros(retshape)*0j
    for inptuple in product(range(n1),range(n2),range(n3),range(n4)):
        rettuple=swaptuple(inptuple,i1,i2)
        retarray[rettuple]=inparray[inptuple]
    return retarray

@autojit
def phasematch(inparray,phasecol,CEParray,rvecarray):
    rhsvecs=swaparray(inparray,0,phasecol)
    (n0,n1,n2,n3)=shape(rhsvecs)
    #print("shape rhsvecs\t"+str(shape(rhsvecs)))
    LHSmat=Linsys(CEParray,rvecarray)
    #print ("shape LHSmat\t"+str(shape(LHSmat)))
    #print("LHSmat\t"+str(LHSmat))
    #print("rhsvecs\t"+str(rhsvecs.reshape((n0,-1))[:,:3]))
    retvecs=solve(LHSmat,rhsvecs.reshape((n0,-1)))
    #print("retvecs\t"+str(retvecs[:,:3]))
    retvecs=reshape(retvecs,(n0,n1,n2,n3))
    return swaparray(retvecs,0,phasecol)

def fftkvec(nfft):
    #return vector with units n*kir representing how many units of kir
    #the total diagram picks up
    dt=1.#equal to (2*pi)/(2*pi), since total phase change is 2*pi
    return list(-floor(nfft/2.)+array(range(nfft)))

def fftdipfreqs(tvec):
    ntvec=len(tvec)
    dt=tvec[1]-tvec[0]
    retvec=fft.fftfreq(ntvec,dt)*2*pi
    retvec.sort()
    return retvec

#################################################
def printphasematchkeys(keyfilename):
    printindx=0
    f=open(phasematchdirstr+keyfilename,'w')
    f.write("#indx\ttxuv\tkxuv\n")
    for (i,j) in product(range(ntxuv),range(nCEPxuv)):
        f.write("\t".join(map(str,[printindx,XUVtvec[i],XUVcepvec[j]]))+"\n")
        printindx+=1
    f.close()

def arraytofile(xvec,yvec,zarray,filename):
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    #print("shape zarray\t"+str(shape(zarray)))
    #print("shape xvec\t"+str(shape(xvec)))
    #print("shape yvec\t"+str(shape(yvec)))
    for i in range(n):
        for j in range(m):
            outfile.write(str(xvec[i])+"\t"+str(real(yvec[j]))+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()
            
#printphasematchedarrays(kftinvfreqarray,kIRvec,fftfreqvec,'kIR_vs_freq.dat')
#def printphasematchedarrays(array,xvec,yvec,dirname,filename):
#    (n1,n2,n3,n4)=shape(array)
#    printindx=0
#    for (i,j) in product(range(n1),range(n2)):
#        subprocess.call(["mkdir",dirname+str(printindx)])
#        arraytofile(xvec,yvec,array[i,j,:,:],\
#                    dirname+str(printindx)+"/"+filename)
#        printindx+=1
def printphasematchedarrays(array,xvec,yvec,dirname,filename):
    (n0,n1,n2,n3)=shape(array)
    i=0
    print("n1\t"+str(n1))
    print("warray",str(warray))
    for j in range(n1):
        subprocess.call(["mkdir",dirname+str(warray[j])])
        arraytofile(xvec[-n2:],yvec[-n3:],array[i,j,:,:],\
                   dirname+str(warray[j])+"/"+filename)
#
#def nDFFT(canglediparray,axislist):
#    cdiparray=copy(canglediparray)
#    fftdiparray=fft.fftn(cdiparray,axes=axislist)
#    #fftdiparray=fft.fftshift(fftdiparray,axes=axislist[1:])
#    return fftdiparray

def nDFFT(canglediparray,axislist, shiftaxes=False, inv=False):
    if(inv):
        fftdiparray=fft.ifftn(canglediparray,axes=axislist)
    else:
        fftdiparray=fft.fftn(canglediparray,axes=axislist)
    if(shiftaxes):
        if(inv):
            fftdiparray=fft.ifftshift(fftdiparray,axes=axislist)
        else:
            fftdiparray=fft.fftshift(fftdiparray,axes=axislist)
    return fftdiparray

def rnDFFT(canglediparray,axislist):
    #shape is (ndtxuv,nCEPxuv,nCEPIR,ndt)
    rdiparray=real(canglediparray)
    fftdiparray=fft.rfftn(rdiparray,axes=axislist)
    #fftdiparray=fft.fftshift(fftdiparray,axes=axislist[1:-1])
    return fftdiparray



def readdiparray(inpdipfilestr,arrayshape,dipcol=2):#2*dipcol=real part, 2*dipcol+1 = imag part
    #returns an array of shape (ntxuv,nCEPxuv,nCEPIR,nt)
    dipfilestr="/"+inpdipfilestr
    if(os.path.isdir("0/Dat")):
        dipfilestr="/Dat"+dipfilestr
    diparraylist=[]
    ncalcs=np.product(list(arrayshape))
    #print("ncalcs\t"+str(ncalcs))
    for i in range(ncalcs):
        diparraylist.append(readarray(i,dipfilestr,False))
        #rvals=True means that only real component of dipole is read

    #make a new array consisting of only the columns of interest
    tvec, diparray=makediparray_raman(dipcol, arrayshape, diparraylist)
    return tvec,diparray


def readanglediparray(thetapol, phipol, arrayshape, dipcol=2):
    #returns the component of the calculated dipole parallel to a given
    #polarization direction
    tvec,zdiparray=readdiparray("ZDipoleexpect.Dat", arrayshape, dipcol)
    if(thetapol==0):
        anglediparray=zdiparray
    else:
        tvec,xdiparray=readdiparray("XDipoleexpect.Dat", arrayshape, dipcol)
        tvec,ydiparray=readdiparray("YDipoleexpect.Dat", arrayshape, dipcol)
        anglediparray=cos(thetapol)*zdiparray+\
                sin(thetapol)*cos(phipol)*xdiparray+\
                sin(thetapol)*sin(phipol)*ydiparray
    return tvec,anglediparray

#################################################
#polarization theta and phi can be specified as command line arguments
parser=argparse.ArgumentParser()
parser.add_argument('-theta',action='store',dest="theta_deg",type=float,
                    default=0.,help='polarization angle theta in degrees')
parser.add_argument('-phi',action='store',dest="phi_deg",type=float,
                    default=0.,help='polarization angle phi in degrees')
parsedarguments=parser.parse_args()
thetapol=parsedarguments.theta_deg/180.*pi
phipol=parsedarguments.phi_deg/180*pi
print("polarization angle\t"+ str(parsedarguments.theta_deg)+
      "\t"+str(parsedarguments.phi_deg))

#read calculation parameters from file written by setup script
dtdownvec, dtstreakvec, CEPupvec, CEPdownvec, CEPstreakvec=readparameterkey_raman('parameterkey.txt')
ndtdown=len(dtdownvec)
ndtstreak=len(dtstreakvec)
nCEPup=len(CEPupvec)
nCEPdown=len(CEPdownvec)
nCEPstreak=len(CEPstreakvec)
arrayshape=(ndtdown,ndtstreak,nCEPup,nCEPdown,nCEPstreak)
#ncalcs=ntxuv*nCEPxuv*nCEPir
#print("nCEPxuv, nCEPir, ncalcs\t"+str(nCEPxuv)+"\t"+str(nCEPir)+"\t"+str(ncalcs))
phasematchdirstr="phasematched/"
subprocess.call(["mkdir",phasematchdirstr])
#rvec=[0,1]#want to solve for 0*kxuv and 1*kxuv components
#read dipoles from file
tvec,anglediparray=readanglediparray(thetapol, phipol, arrayshape, dipcol=2)
#read pulses from file
tvec,angleEarray=readanglediparray(thetapol, phipol, arrayshape, dipcol=3)

################################################################################
#take fourier transforms

n_vs_tarray=nDFFT(anglediparray,[2,3,4],shiftaxes=True)
n_vs_warray=nDFFT(n_vs_tarray,[5],shiftaxes=True)


nupvec=fftkvec(nCEPup)
ndownvec=fftkvec(nCEPdown)
nstreakvec=fftkvec(nCEPstreak)
wvec=fftdipfreqs(tvec)

wvec.dump(phasematchdirstr+"wvec.pkl")
tvec.dump(phasematchdirstr+"tmeasurevec.pkl")
array(nupvec).dump(phasematchdirstr+"nupvec.pkl")
array(ndownvec).dump(phasematchdirstr+"ndownvec.pkl")
array(nstreakvec).dump(phasematchdirstr+"nstreakvec.pkl")
n_vs_tarray.dump(phasematchdirstr+"n_vs_tarray.pkl")
n_vs_warray.dump(phasematchdirstr+"n_vs_warray.pkl")
dtdownvec.dump(phasematchdirstr+"dtdownvec.pkl")
dtstreakvec.dump(phasematchdirstr+"dtstreakvec.pkl")
CEPupvec.dump(phasematchdirstr+"CEPupvec.pkl")
CEPdownvec.dump(phasematchdirstr+"CEPdownvec.pkl")
CEPstreakvec.dump(phasematchdirstr+"CEPstreakvec.pkl")

#kIRvstarray=nDFFT(anglediparray,[1,2],shiftaxes=True)
#kIRvswarray=nDFFT(anglediparray,[1,2,3],shiftaxes=True)
#phasevstarray=anglediparray#nDFFT(anglediparray,[1],shiftaxes=True)
#phasevswarray=nDFFT(anglediparray,[3],shiftaxes=True)
#
##fourier transforms of input pulses
#Earray=angleEarray
#Ewarray=nDFFT(Earray,[3],shiftaxes=True)
#
##make kIR, frequency vectors for use in printing
#kIRvec=fftkvec(nCEPir)
#fftfreqvec=fftdipfreqs(tvec)
#
#printphasematchedarrays(kIRvstarray,kIRvec,tvec,phasematchdirstr,
#                        'kIR_vs_t.dat')
#printphasematchedarrays(kIRvswarray,kIRvec,fftfreqvec,phasematchdirstr,
#                        'kIR_vs_freq.dat')
#printphasematchedarrays(phasevstarray,IRcepvec,tvec,phasematchdirstr,
#                        'phase_vs_t.dat')
#printphasematchedarrays(phasevswarray,IRcepvec,fftfreqvec,phasematchdirstr,
#                        'phase_vs_freq.dat')
##pickle results
#kIRvstarray.dump(phasematchdirstr+"kIRvstarray.pkl")
#kIRvswarray.dump(phasematchdirstr+"kIRvswarray.pkl")
#phasevstarray.dump(phasematchdirstr+"phasevstarray.pkl")
#phasevswarray.dump(phasematchdirstr+"phasevswarray.pkl")
#kIRvec.dump(phasematchdirstr+"kIRvec.pkl")
#tvec.dump(phasematchdirstr+"tmeasurevec.pkl")
#fftfreqvec.dump(phasematchdirstr+"woutvec.pkl")
#IRcepvec.dump(phasematchdirstr+"IRcepvec.pkl")
#XUVtvec.dump(phasematchdirstr+"deltatvec.pkl")
#
#Earray.dump(phasematchdirstr+"phaseEarray.pkl")
#Ewarray.dump(phasematchdirstr+"phaseEwarray.pkl")
