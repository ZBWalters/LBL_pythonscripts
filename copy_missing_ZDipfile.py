import glob
import shutil
from math import floor
from os.path import isdir

#####
#Sometimes, for reasons unknown, the MCTDHF code doesn't run for
#particular sets of inputs in a streaking calculation.  This script
#exploits the symmetry that, for the component of the dipole which
#absorbs 1 xuv photon, it must also absorb an even number of IR
#photons.  Thus, copying the dipoles from the inputs corresponding to
#phiprime=phi+pi (for both phixuv=0 and phixuv=pi) preserves the
#difference in the phase matching condition, and allows the
#calculation of the streaking spectra.

#Note that this is not a replacement for simply running the code again
#for the sets of inputs that didn't go through the first time.  (In
#particular, this will screw up the phase matched component which does
#not correspond to absorbtion of an xuv photon).  However, it will
#give the appropriate spectra of interest without requiring several
#hours wait while the calculations run for the missing input sets.

datfilestr=""
if(isdir("0/Dat")):
    datfilestr="/Dat"

def dirname(filestr):
    return filestr.split("/")[0]+"/"

def fftpartner(n,nfft):
    return floor(n/nfft)*nfft+(n+nfft/2)%(nfft)

def copypartner(dirname):
    dirint=int(dirname[:-1])
    dirpartner=fftpartner(dirint,nfft)
    print("copying from "+str(int(dirpartner))+"/ to "+dirname)
    shutil.copy(str(int(dirpartner))+datfilestr+"/ZDipoleexpect.Dat",dirname+datfilestr)
    shutil.copy(str(int(dirpartner))+datfilestr+"/XDipoleexpect.Dat",dirname+datfilestr)
    shutil.copy(str(int(dirpartner))+datfilestr+"/YDipoleexpect.Dat",dirname+datfilestr)

zdiplist=glob.glob("**"+datfilestr+"/ZD*exp*.Dat")
zdipdirlist=list(map(dirname,zdiplist))
dirlist=glob.glob("[0-9]*/")

nfft=int(len(dirlist)/2)#24
print("set(dirlist)-set(zdipdirlist)\t"+str(set(dirlist)-set(zdipdirlist)))

for dirname in set(dirlist)-set(zdipdirlist):
    copypartner(dirname)


    #also copy versions with different phi_xuv (so that difference is preserved)
    dirint=int(dirname[:-1])
    dirint2=int(((floor(dirint/nfft)+1)%2)*nfft)+(dirint%nfft)
    dirint2name=str(dirint2)+"/"
    copypartner(dirint2name)
