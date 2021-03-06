import glob
import argparse
from numpy import *
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

def kIRtransformdiparray(diparray):
    retarray=fft.fft(diparray,axis=-2)
    retarray=fft.fftshift(retarray,axes=(-2))
    return retarray

def FFTdiparray(diparray):
    retarray=fft.fft(diparray,axis=-1)
    retarray=fft.fftshift(retarray,axes=(-1))
    return retarray


def readparameterkey(inpfilename):
    #read parameters for the calculation from the file written by the
    #setup script
    inparray=genfromtxt(inpfilename,comments="#")
    print("shape inparray\t"+str(shape(inparray)))
    txuvvec=sort(list(uniquevals(inparray[:,1])))
    XUVcepvec=sort(list(uniquevals(inparray[:,2])))
    IRcepvec=sort(list(uniquevals(inparray[:,3])))
    return txuvvec, XUVcepvec,IRcepvec

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

def printphasematchedarrays(array,xvec,yvec,dirname,filename):
    (n0,n1,n2,n3)=shape(array)
    i=0
    print("n1\t"+str(n1))
    print("warray",str(warray))
    for j in range(n1):
        subprocess.call(["mkdir",dirname+str(warray[j])])
        arraytofile(xvec[-n2:],yvec[-n3:],array[i,j,:,:],\
                   dirname+str(warray[j])+"/"+filename)

def phasematchedarraystofiles(array,dtvec,xvec,yvec,dirname,filename):
    (n0,n1,n2,n3)=shape(array)
    nvec=range(n1)
    for npr in range(n1):#npr=number of probe photons
        nstr="npr_"+str(warray[npr])+"/"
        subprocess.call(["mkdir",dirname+nstr[:-1]])
        for m in range(n0):
            dt=dtvec[m]
            dtstr=str(dt)+"_aut/"
            subprocess.call(["mkdir",dirname+nstr+dtstr[:-1]])
            arraytofile(xvec[-n2:],yvec[-n3:],array[m,npr,:,:],
                        dirname+nstr+dtstr+"/"+filename)

def nDFFT(canglediparray,axislist):
    cdiparray=real(canglediparray)
    fftdiparray=fft.fftn(cdiparray,axes=axislist)
    #fftdiparray=fft.fftshift(fftdiparray,axes=axislist[1:])
    return fftdiparray

def rnDFFT(canglediparray,axislist):
    #shape is (ndtxuv,nCEPxuv,nCEPIR,ndt)
    rdiparray=real(canglediparray)
    fftdiparray=fft.rfftn(rdiparray,axes=axislist)
    #fftdiparray=fft.fftshift(fftdiparray,axes=axislist[1:-1])
    return fftdiparray



def readdiparray(inpdipfilestr):
    #returns an array of shape (ntxuv,nCEPxuv,nCEPIR,nt)
    dipfilestr="/"+inpdipfilestr
    if(os.path.isdir("0/Dat")):
        dipfilestr="/Dat"+dipfilestr
    diparraylist=[]
    for i in range(ncalcs):
        diparraylist.append(readarray(i,dipfilestr,False))
        #rvals=True means that only real component of dipole is read

    dipcol=2#column containing the dipoles of interest
    #corresponds to columns 4 and 5 in the unmerged output files
    tvec,diparray=makediparray(dipcol,ntxuv,nCEPxuv,nCEPir,diparraylist)
    return tvec,diparray


def readanglediparray(thetapol,phipol):
    #returns the component of the calculated dipole parallel to a given
    #polarization direction
    tvec,zdiparray=readdiparray("ZDipoleexpect.Dat")
    if(thetapol==0):
        anglediparray=zdiparray
    else:
        tvec,xdiparray=readdiparray("XDipoleexpect.Dat")
        tvec,ydiparray=readdiparray("YDipoleexpect.Dat")
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
XUVtvec, XUVcepvec, IRcepvec=readparameterkey('parameterkey.txt')
ntxuv=len(XUVtvec)
nCEPxuv=len(XUVcepvec)
nxuvby2=int(floor(nCEPxuv/2))
warray=list(range(-nxuvby2,-nxuvby2+nCEPxuv,1))
nCEPir=len(IRcepvec)
ncalcs=ntxuv*nCEPxuv*nCEPir
print("nCEPxuv, nCEPir, ncalcs\t"+str(nCEPxuv)+"\t"+str(nCEPir)+"\t"+str(ncalcs))
phasematchdirstr="phasematched/"
subprocess.call(["mkdir",phasematchdirstr])
tvec,anglediparray=readanglediparray(thetapol,phipol)

#take fourier transforms (note that because diple is real valued, shape of
#returned arrays will not equal shape of input array.  The missing terms can be
#found by symmetry)

kIRvstarray=nDFFT(anglediparray,[1,2])
kIRvstarray=fft.fftshift(kIRvstarray,[1,2])
kIRvswarray=nDFFT(anglediparray,[1,2,3])
kIRvswarray=fft.fftshift(kIRvswarray,[1,2,3])
phasevstarray=nDFFT(anglediparray,[1])
phasevstarray=fft.fftshift(phasevstarray,[1])
phasevswarray=nDFFT(anglediparray,[1,3])
phasevswarray=fft.fftshift(phasevswarray,[1,3])
#make kIR, frequency vectors for use in printing
kIRvec=fftkvec(nCEPir)
fftfreqvec=fftdipfreqs(tvec)

#Write pickled arrays to files for retrieval by python programs
kIRvec.dump(phasematchdirstr+"nsvec.pkl")
fftfreqvec.dump(phasematchdirstr+"woutvec.pkl")
XUVtvec.dump(phasematchdirstr+"deltatvec.pkl")
tvec.dump(phasematchdirstr+"tmeasurevec.pkl")
IRcepvec.dump(phasematchdirstr+"phasevec.pkl")
array(warray).dump(phasematchdirstr+"nprvec.pkl")
kIRvstarray.dump(phasematchdirstr+"deltat_npr_ns_tmeasure_array.pkl")
kIRvswarray.dump(phasematchdirstr+"deltat_npr_ns_wout_array.pkl")
phasevstarray.dump(phasematchdirstr+"deltat_npr_phase_tmeasure_array.pkl")
phasevswarray.dump(phasematchdirstr+"deltat_npr_phase_wout_array.pkl")

#Write array cuts to text files for possible retrieval by non-python programs
phasematchedarraystofiles(kIRvstarray, XUVtvec, kIRvec, tvec, phasematchdirstr,
                          'kIR_vs_t.dat')
phasematchedarraystofiles(kIRvswarray, XUVtvec, kIRvec, fftfreqvec, phasematchdirstr,
                          'kIR_vs_freq.dat')
phasematchedarraystofiles(phasevstarray, XUVtvec, IRcepvec, tvec, phasematchdirstr,
                          'phase_vs_t.dat')
phasematchedarraystofiles(phasevswarray, XUVtvec, IRcepvec, fftfreqvec, phasematchdirstr,
                          'phase_vs_freq.dat')
