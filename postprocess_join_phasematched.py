import glob
import re
from numpy import *
import subprocess
from itertools import product
from sys import argv
import pickle

def natural_sort(l):
    #sort a list such that numbers appear in alphanumeric order rather than
    #the usual unix ordering
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def datfromfile(filename):
    #read data from a file
    loadarray=loadtxt(filename)
    xdat=loadarray[:,0]
    ydat=loadarray[:,1]
    zdat=loadarray[:,2]+1j*loadarray[:,3]
    return [xdat,ydat,zdat]

def uniquevals(inplist):
#    #unique elements of a list (not guaranteed to preserve original
#    #list's order)
    return sorted(list(set(inplist)))
#    retlist=[]
#    for elt in inplist:
#        if (not (elt in retlist)):
#            retlist.append(elt)
#    retlist.sort()
#    return retlist



def dattoarrays(xdat,ydat,zdat):
    #convert one dimensional dat arrays to two dimensional arrays
    n=len(uniquevals(xdat))
    m=len(uniquevals(ydat))
    xarray=reshape(xdat,(n,m))
    yarray=reshape(ydat,(n,m))
    yvec=yarray[0,:]
    zarray=reshape(zdat,(n,m))
    return [xarray,yarray,zarray]

def arraystodat(xarray,yarray,zarray):
    return xarray.flatten(),yarray.flatten(),zarray.flatten()

def arraysfromfile(filename):
    xdat,ydat,zdat=datfromfile(filename)
    xarray,yarray,zarray=dattoarrays(xdat,ydat,zdat)
    return xarray,yarray,zarray

def readfilelist(dirlist,phasematchdirstr,dirstr,filestr):
    retlist=[]
    for i in range(len(dirlist)):
        xarray,yarray,zarray=arraysfromfile(dirlist[i]+phasematchdirstr+\
                                            dirstr+filestr)
        retlist.append(zarray)
    return xarray,yarray,array(retlist)

def writearraycuts(xvec,yvec,zvec,datarray,multiDdirstr,outfilename):
    print("writing multidimensional array, shape="+str(shape(datarray)))
    subprocess.call(["mkdir",multiDdirstr])
    for j in range(len(yvec)):
        tmpdirstr=multiDdirstr+str(j)+"/"
        subprocess.call(["mkdir",tmpdirstr])
        filename=tmpdirstr+outfilename
        f=open(filename,'w')
        f.write("#y value\t"+str(yvec[j])+"\n")
        f.write("#xval \t zval\n")
        for i,k in product(range(len(xvec)),range(len(zvec))):
            f.write(str(xvec[i])+"\t"+str(zvec[k])+"\t"+
                    str(real(datarray[i,j,k]))+"\t"+
                    str(imag(datarray[i,j,k]))+"\n")
        f.close()

Hrt=27.21#Hartree in eV
aut=24.2
#########################################################
if(len(argv)>2):
    dw0=float(argv[-1])/Hrt#.0565*Hrt
else:
    dw0=1./Hrt
print("dw0 = "+str(dw0*Hrt)+" eV")
phasematchdirstr="phasematched/"
dirstr="1/"
multiDdirstr="phasematched_multidimensional/"
dirlist=natural_sort(glob.glob("results*/"))
print("dirlist\t"+str(dirlist))
warray=array(list(range(1,len(dirlist)+1)))*dw0

#read and join phase_vs_freq.dat
filestr="phase_vs_freq.dat"
xdat,ydat,zdatarray=readfilelist(dirlist,phasematchdirstr,dirstr,filestr)
xvec=uniquevals(xdat.flatten())
yvec=uniquevals(ydat.flatten())
writearraycuts(warray,xvec,yvec,zdatarray,multiDdirstr,"freq_vs_freq_phasecut.dat")
#write pickled array to a file
zdatarray.dump(multiDdirstr+"freq_vs_phase_vs_freq.pkl")
array(yvec).dump(multiDdirstr+"woutvec.pkl")
array(xvec).dump(multiDdirstr+"phasevec.pkl")
warray.dump(multiDdirstr+"winpvec.pkl")

#read and join phase_vs_t.dat
filestr="phase_vs_t.dat"
xdat,ydat,zdatarray=readfilelist(dirlist,phasematchdirstr,dirstr,filestr)
xvec=uniquevals(xdat.flatten())
yvec=uniquevals(ydat.flatten())
writearraycuts(warray,xvec,yvec,zdatarray,multiDdirstr,"freq_vs_t_phasecut.dat")
#write pickled array to a file
zdatarray.dump(multiDdirstr+"freq_vs_phase_vs_t.pkl")
array(yvec).dump(multiDdirstr+"tvec.pkl")

#read and join kIR_vs_freq.dat
filestr="kIR_vs_freq.dat"
xdat,ydat,zdatarray=readfilelist(dirlist,phasematchdirstr,dirstr,filestr)
xvec=uniquevals(xdat.flatten())
yvec=uniquevals(ydat.flatten())
writearraycuts(warray,xvec,yvec,zdatarray,multiDdirstr,"freq_vs_freq_kIRcut.dat")
#write pickled array to a file
zdatarray.dump(multiDdirstr+"freq_vs_kIR_vs_freq.pkl")
array(xvec).dump(multiDdirstr+"kIRvec.pkl")

#read and join kIR_vs_t.dat
filestr="kIR_vs_t.dat"
xdat,ydat,zdatarray=readfilelist(dirlist,phasematchdirstr,dirstr,filestr)
xvec=uniquevals(xdat.flatten())
yvec=uniquevals(ydat.flatten())
writearraycuts(warray,xvec,yvec,zdatarray,multiDdirstr,"freq_vs_t_kIRcut.dat")
#write pickled array to a file
zdatarray.dump(multiDdirstr+"freq_vs_kIR_vs_t.pkl")

