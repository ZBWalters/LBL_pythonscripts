import glob
import argparse
from numpy import *
from numpy.linalg import solve
from itertools import *
import subprocess
import shutil
import os.path
from numba import jit, autojit
#import matplotlib.pyplot as plt

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


def arrayindx(inplist,shapelist):
    #print("inplist\t"+str(inplist))
    #print("shapelist\t"+str(shapelist))
    nvars=len(inplist)
    multfact=1
    retval=0
    for i in range(nvars-1,-1,-1):
        retval+=multfact*inplist[i]
        multfact*=shapelist[i]
    #print("retval\t"+str(retval))
    return retval




#    for (itxuv,iCEPxuv,iCEPir) in \
#        product(range(ntxuv),range(nCEPxuv),range(nCEPir)):
#        indx=itxuv*(nCEPxuv*nCEPir)+iCEPxuv*(nCEPir)+iCEPir
#        #print("indx\t"+str((itxuv,iCEPxuv,iCEPir))+"\t"+str(indx))
#        retarray[itxuv,iCEPxuv,iCEPir,:]=arraylist[indx][dipcol,:]
    return tvec,retarray

def kIRtransformdiparray(diparray):
    retarray=fft.fft(diparray,axis=-2)
    retarray=fft.fftshift(retarray,axes=(-2))
    return retarray

def FFTdiparray(diparray):
    retarray=fft.fft(diparray,axis=-1)
    retarray=fft.fftshift(retarray,axes=(-1))
    return retarray


def readparameterkey_threepulseraman(inpfilename):
    #read parameters for the calculation from the file written by the
    #setup script
    paramarray=genfromtxt(inpfilename,comments="#")
    print("shape paramarray\t"+str(shape(paramarray)))
    sCEPvec=sort(list(uniquevals(paramarray[:,1])))
    ram1CEPvec=sort(list(uniquevals(paramarray[:,2])))
    ram2CEPvec=sort(list(uniquevals(paramarray[:,3])))
    return sCEPvec, ram1CEPvec, ram2CEPvec, paramarray

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
    return -nfft*dt/2.+dt*array(range(nfft))

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

def nDFFT(canglediparray,axislist, shiftaxes=False):
    cdiparray=real(canglediparray)
    fftdiparray=fft.fftn(cdiparray,axes=axislist)
    if(shiftaxes):
        fftdiparray=fft.fftshift(fftdiparray,axes=axislist)
    return fftdiparray

def rnDFFT(canglediparray,axislist, shiftaxes=False):
    #shape is (ndtxuv,nCEPxuv,nCEPIR,ndt)
    rdiparray=real(canglediparray)
    fftdiparray=fft.rfftn(rdiparray,axes=axislist)
    if(shiftaxes):
        fftdiparray=fft.fftshift(fftdiparray,axes=axislist)
    return fftdiparray

def vecindices(calcindx,veclist):
    tmpvec=paramarray[calcindx,:]
    retlist=[]
    for i in range(len(veclist)):
        retlist.append(list(veclist[i]).index(tmpvec[i+1]))
    return retlist

def makediparray(dipcol,veclist,arraylist):
    (ncols,nt)=shape(arraylist[0])
    tvec=arraylist[0][0,:]
    shapelist=list(map(len,veclist))
    augshapelist=shapelist+[nt]
    retarray=zeros(tuple(augshapelist))*0j
    #print("shape retarray\t"+str(shape(retarray)))
    for i in range(len(arraylist)):
        vecindxlist=vecindices(i,veclist)
        #print("vecindxlist\t"+str(i)+"\t"+str(vecindxlist))
        retarray[vecindxlist[0],vecindxlist[1],vecindxlist[2],:]=arraylist[i][dipcol,:]
#    rangelst=list(map(range,shapelist))
#    for arglist in product(*rangelst):
#        indx=arrayindx(arglist,shapelist)
#        retarray[arglist,:]=arraylist[indx][dipcol,:]
    return tvec, retarray

#def makediparray(dipcol,shapelist,arraylist):
#    (ncols,nt)=shape(arraylist[0])
#    tvec=arraylist[0][0,:]
#    augshapelist=shapelist+[nt]
#    retarray=zeros(tuple(augshapelist))*0j
#    rangelst=list(map(range,shapelist))
#    for arglist in product(*rangelst):
#        indx=arrayindx(arglist,shapelist)
#        retarray[arglist,:]=arraylist[indx][dipcol,:]
#    return tvec, retarray

def readdiparray(inpdipfilestr, veclist):
    #returns an array of shape (ntxuv,nCEPxuv,nCEPIR,nt)
    dipfilestr="/"+inpdipfilestr
    if(os.path.isdir("0/Dat")):
        dipfilestr="/Dat"+dipfilestr
    diparraylist=[]
    for i in range(ncalcs):
        diparraylist.append(readarray(i,dipfilestr,False))
        #rvals=True means that only real component of dipole is read
    #print("shape diparraylist\t"+str(shape(diparraylist)))
    dipcol=2#column containing the dipoles of interest
    #corresponds to columns 4 and 5 in the unmerged output files
    tvec,diparray=makediparray(dipcol,veclist,diparraylist)
    return tvec,diparray


def readanglediparray(thetapol,phipol,veclist):
    #returns the component of the calculated dipole parallel to a given
    #polarization direction
    tvec,zdiparray=readdiparray("ZDipoleexpect.Dat", veclist)
    if(thetapol==0):
        anglediparray=zdiparray
    else:
        tvec,xdiparray=readdiparray("XDipoleexpect.Dat", veclist)
        tvec,ydiparray=readdiparray("YDipoleexpect.Dat", veclist)
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
#XUVtvec, XUVcepvec, IRcepvec=readparameterkey('parameterkey.txt')
sCEPvec, ram1CEPvec, ram2CEPvec, paramarray = \
    readparameterkey_threepulseraman('parameterkey_threepulseraman.txt')
#ntxuv=len(XUVtvec)
#nCEPxuv=len(XUVcepvec)
#nxuvby2=int(floor(nCEPxuv/2))
#warray=list(range(-nxuvby2,-nxuvby2+nCEPxuv,1))
#nCEPir=len(IRcepvec)
#ncalcs=ntxuv*nCEPxuv*nCEPir
nsCEP=len(sCEPvec)
nram1CEP=len(ram1CEPvec)
nram2CEP=len(ram1CEPvec)
nsCEPby2=int(floor(nsCEP/2))
nram1CEPby2=int(floor(nram1CEP/2))
nram2CEPby2=int(floor(nram2CEP/2))
nsarray=array(range(nsCEP))-nsCEPby2
nram1array=array(range(nram1CEP))-nram1CEPby2
nram2array=array(range(nram2CEP))-nram2CEPby2
ncalcs=nsCEP*nram1CEP*nram2CEP
#print("nCEPxuv, nCEPir, ncalcs\t"+str(nCEPxuv)+"\t"+str(nCEPir)+"\t"+str(ncalcs))
print("nsCEP, nram1CEP, nram2CEP, ncalcs\t" + str(nsCEP) + "\t" + str(nram1CEP)
      + "\t" + str(nram2CEP) + "\t" + str(ncalcs))
phasematchdirstr="phasematched/"
subprocess.call(["mkdir",phasematchdirstr])
tvec,anglediparray=readanglediparray(thetapol,phipol,[sCEPvec,ram1CEPvec,ram2CEPvec])

phis_phir1_phir2_tarray=anglediparray
phis_phir1_phir2_warray=nDFFT(phis_phir1_phir2_tarray,axislist=[3],shiftaxes=True)
ns_nr1_nr2_tarray=nDFFT(phis_phir1_phir2_tarray,axislist=[0,1,2],shiftaxes=True)
ns_nr1_nr2_warray=nDFFT(phis_phir1_phir2_tarray,axislist=[0,1,2,3],shiftaxes=True)
fftfreqvec=fftdipfreqs(tvec)

#dump phase matched components to phasematchdir
phis_phir1_phir2_tarray.dump(phasematchdirstr+"phis_phir1_phir2_tarray.pkl")
phis_phir1_phir2_warray.dump(phasematchdirstr+"phis_phir1_phir2_warray.pkl")
ns_nr1_nr2_tarray.dump(phasematchdirstr+"ns_nr1_nr2_tarray.pkl")
ns_nr1_nr2_warray.dump(phasematchdirstr+"ns_nr1_nr2_warray.pkl")
nsarray.dump(phasematchdirstr+"nsvec.pkl")
nram1array.dump(phasematchdirstr+"nram1vec.pkl")
nram2array.dump(phasematchdirstr+"nram2vec.pkl")
array(sCEPvec).dump(phasematchdirstr+"sCEPvec.pkl")
array(ram1CEPvec).dump(phasematchdirstr+"ram1CEPvec.pkl")
array(ram2CEPvec).dump(phasematchdirstr+"ram2CEPvec.pkl")
array(tvec).dump(phasematchdirstr+"tmeasurevec.pkl")
array(fftfreqvec).dump(phasematchdirstr+"woutvec.pkl")
