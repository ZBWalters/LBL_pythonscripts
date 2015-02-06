from numpy import *
from numpy.fft import *
from scipy import *
import glob

def correctt1t3offset(twodfreqarray):
    #adjust Fourier Transform array calculated with 2nd time measured
    #wrt to pulse 1 to a fourier transform array calculated with 2nd
    #time measured wrt t3 using fourier shift theorem
    rettwodfreqarray=twodfreqarray*0j
    (n,m)=shape(twodfreqarray)
    for j in range(m):
        shiftint=int(round((j/(m)-.5)*n))
        for i in range(n):
            rettwodfreqarray[mod(i+shiftint,n),j]=twodfreqarray[i,j]
    return rettwodfreqarray

def datfromfile(filename):
    loadarray=loadtxt(filename)
    xdat=loadarray[:,0]
    ydat=loadarray[:,1]
    zdat=loadarray[:,2]+1j*loadarray[:,3]
    return [xdat,ydat,zdat]

def dattoarrays(xdat,ydat,zdat):
    n=len(uniquevals(xdat))
    m=len(uniquevals(ydat))
    xarray=reshape(xdat,(n,m))
    yarray=reshape(ydat,(n,m))
    yvec=yarray[0,:]
    zarray=reshape(zdat,(n,m))
    return [xarray,yarray,zarray]

def arraytofile(xarray,yarray,zarray,filename):
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    for i in range(n):
        for j in range(m):
            outfile.write(str(xarray[i])+"\t"+str(yarray[j])+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

def uniquevals(inplist):
    retlist=[]
    for elt in inplist:
        if (not (elt in retlist)):
            retlist.append(elt)
    retlist.sort()
    return retlist

def correctarrayfile(inpfilename):
    [xdat,ydat,zdat]=datfromfile(inpfilename)
    [xarray,yarray,zarray]=dattoarrays(xdat,ydat,zdat)
    xvec=uniquevals(xdat)
    yvec=uniquevals(ydat)
    zarray=correctt1t3offset(zarray)
    return xvec,yvec,zarray

def savecorrectarrayfile(inpfilename):
    xvec,yvec,zarray=correctarrayfile(inpfilename)
    tvec0,yvec0,ifftzarray0=invertfft(0,xvec,yvec,zarray)
    xvec1,tvec1,ifftzarray1=invertfft(1,xvec,yvec,zarray)
    arraytofile(xvec,yvec,zarray,inpfilename[:-4]+"_shift.dat")
    arraytofile(tvec0,yvec0,ifftzarray0,inpfilename[:-4]+"_t0f1.dat")
    arraytofile(xvec1,tvec1,ifftzarray1,inpfilename[:-4]+"_f0t1.dat")

def correctfreqarrays():
    filelist=glob.glob("twodfreqarray.[0-9].dat")+glob.glob("twodfreqarray.[0-9][0-9].dat")
    for filename in filelist:
        print("correcting "+filename)
        savecorrectarrayfile(filename)

def invertfft(indx,xvec,yvec,zarray):
    ifftzarray=ifft(zarray,axis=indx)
    #ifftshift(ifftzarray,axes=indx)
    if(indx==0):
        tvec=(fftfreq(len(xvec),xvec[1]-xvec[0]))
        retlist=tvec,yvec,ifftzarray
    if(indx==1):
        tvec=(fftfreq(len(yvec),yvec[1]-yvec[0]))
        retlist=xvec,tvec,ifftzarray
    return retlist


############################################################
correctfreqarrays()
