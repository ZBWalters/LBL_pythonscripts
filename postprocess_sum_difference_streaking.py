from numpy import *
#import glob
#from numba import jit, autojit
import subprocess
#import os.path
import pickle
from scipy.fftpack import dct, idct

def readloopkey():
    inpfile=open('./loopkey.txt','r')
    namelist=[]
    veclist=[]
    header=inpfile.readline()
    for line in inpfile:
        splitline=line.split("\t")
        namelist.append(splitline[0])
        veclist.append([float(item) for item in splitline[1:]])

    return namelist, veclist

def readcolumnfromfile(filename, rcol=0, imcol=None):
    if(imcol!=None):
        usecols=(rcol,imcol)
    else:
        usecols=(rcol)

    coldata=genfromtxt(filename, usecols=usecols, comments="#")

    if(imcol!=None):
        retcol= coldata[:,0]+1j*coldata[:,1]
    else:
        retcol=coldata
    return retcol

def readdipoletimes(filename="0/Dat/ZDipoleexpect.Dat"):
    return readcolumnfromfile(filename,rcol=0)

def readdipolefromfile(filename, rvals=False, colindx=2):
#read real and imaginary components of dipole from file
    rcol=2*colindx-1
    imcol=2*colindx
    if(rvals):
        imcol=None

    return readcolumnfromfile(filename, rcol=rcol, imcol=imcol)

def readpulsefromfile(filename, rvals=False, colindx=3):
    return readdipolefromfile(filename=filename, rvals=rvals, colindx=colindx)

def readdipolefiles(filenamestr="Dat/ZDipoleexpect.Dat", rvals=False, colindx=2):
    #first, read in vectors to be looped over
    namelist, veclist=readloopkey()
    lenlist=[len(item) for item in veclist]
    #each set of indices will read in a vector of length ndip
    tvec=readdipoletimes()
    nindices=product(lenlist)
    ndip=len(tvec)
    nelts=nindices*ndip#total number of elements to be read
    retarray=zeros(nelts, dtype='complex', order='C')#C style ordering=least significant last
    #print("shape retarray\t"+str(shape(retarray)))
    #print("nindices, ndip\t"+str((nindices,ndip)))
    for i in range(nindices):
        istr=str(i)+"/"
        #print("itstr\t"+istr)
        dipvec=readdipolefromfile(filename=istr+filenamestr, rvals=rvals,
                                   colindx=colindx)
        #print("shape dipvec\t"+str(shape(dipvec)))
        retarray[i*ndip:(i+1)*ndip]=dipvec
    #now reshape retarray to make it a multidimensional array
    lenlist.append(ndip)
    retarray=retarray.reshape(tuple(lenlist))
    print("lenlist\t"+str(lenlist))
    return retarray

def readpulsefiles(filenamestr="Dat/ZDipoleexpect.Dat", rvals=False, colindx=3):
    return readdipolefiles(filenamestr=filenamestr, rvals=rvals, colindx=colindx)



################################################################################
#routines for processing data

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

def nDDCT(canglediparray, type=2, axis=-1, inv=False):
#call discrete cosine transfer library with same calling signature as nDFFT
    if(inv):
        rdctdiparray=idct(real(canglediparray), type=type, axis=axis)
        imdctdiparray=idct(imag(canglediparray), type=type, axis=axis)
        dctdiparray=rdctdiparray+1j*imdctdiparray
    else:
        rdctdiparray=dct(real(canglediparray), type=type, axis=axis)
        imdctdiparray=dct(imag(canglediparray), type=type, axis=axis)
        dctdiparray=rdctdiparray+1j*imdctdiparray
    return dctdiparray

def fftdipfreqs(tvec):
    ntvec=len(tvec)
    dt=tvec[1]-tvec[0]
    retvec=fft.fftfreq(ntvec,dt)*2*pi
    retvec.sort()
    return retvec

def nvec(ntot):
    nmin=-int(floor(ntot/2))
    nmax=nmin+ntot
    return list(range(nmin,nmax))

################################################################################
#file writing routine
def pickledump(obj, filename):
#    tmpfile=open(filename,'w')
#    pickle.dump(obj,tmpfile)
#    tmpfile.close()
    obj.dump(filename)

################################################################################
#main program

#list which axes need to be fourier transformed over and which need a discrete cosine transform
ftaxislist=[1,2,4]#list(range(1,nloops))#fourier transform over all angles except the last 
cosaxis=3#discrete cosine transform over last angle


namelist, veclist=readloopkey()
nloops=len(namelist)
veclenlist=[len(item) for item in veclist]
ftveclist=[nvec(length) for length in veclenlist]
ftveclist[cosaxis]=list(range(veclenlist[cosaxis]))
print("ftveclist\t"+str(ftveclist))

zdiparray=readdipolefiles(filenamestr="Dat/ZDipoleexpect.Dat", rvals=False, colindx=2)

pulsezarray=readdipolefiles(filenamestr="Dat/ZDipoleexpect.Dat", rvals=False, colindx=3)
tvec=readdipoletimes()
namelist.append('tvec')
veclist.append(tvec)

#perform fourier transformations

ftzdiparray=nDFFT(zdiparray, shiftaxes=True, axislist=ftaxislist)
ftzdiparray=nDDCT(ftzdiparray, axis=cosaxis)

ftpulsezarray=nDFFT(pulsezarray, shiftaxes=True, axislist=ftaxislist)
ftpulsezarray=nDDCT(ftpulsezarray, axis=cosaxis)

freqvec=fftdipfreqs(tvec)
ftveclist.append(freqvec)

#pickle results and save
resultdir="phasematched/"
subprocess.call(["mkdir",resultdir])

freqvec.dump(resultdir+"freqvec.pkl")
zdiparray.dump(resultdir+"zdiparray.pkl")


pulsezarray.dump(resultdir+"pulsezarray.pkl")

ftzdiparray.dump(resultdir+"ftzdiparray.pkl")

ftpulsezarray.dump(resultdir+"ftpulsezarray.pkl")


array(namelist).dump(resultdir+"namelist.pkl")
array(veclist).dump(resultdir+"veclist.pkl")
array(ftaxislist).dump(resultdir+"ftaxislist.pkl")#list of axes over which ft has been performed
array(ftveclist).dump(resultdir+"ftveclist.pkl")#index vectors for transformed array
