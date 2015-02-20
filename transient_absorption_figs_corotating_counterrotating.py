from numpy import *
import sys
import os
sys.path.append(os.path.expanduser("~/pythonscripts/"))
from twoDplot import *
from numpy.fft import *

#read pickled files
def readpickledfiles():
    freqvec=load("freqvec.pkl")
    ftaxislist=load("ftaxislist.pkl")
    ftpulsexarray=load("ftpulsexarray.pkl")
    ftpulsezarray=load("ftpulsezarray.pkl")
    ftveclist=load("ftveclist.pkl")
    ftxdiparray=load("ftxdiparray.pkl")
    ftzdiparray=load("ftzdiparray.pkl")
    namelist=load("namelist.pkl")
    pulsexarray=load("pulsexarray.pkl")
    pulsezarray=load("pulsezarray.pkl")
    veclist=load("veclist.pkl")
    xdiparray=("xdiparray.pkl")
    zdiparray=("zdiparray.pkl")
    return [freqvec, ftaxislist, ftpulsexarray, ftpulsezarray, ftveclist,
            ftxdiparray, ftzdiparray,  namelist, pulsexarray, pulsezarray,
            veclist, xdiparray, zdiparray]

def indxlist(inplist, vallist):
    retlist=[]
    for i in range(len(vallist)):
        retlist.append(inplist[i].index(vallist[i]))
    return retlist


def transientabsorption(diparray,Earray):
    (n0,n1)=shape(diparray)
    retarray=zeros((n0,n1))
    for i,j in product(range(n0), range(n1)):
        retarray[i,j]=2*imag(diparray[i,j] * conjugate(Earray[i,j])) /pow(abs(Earray[i,j]),2)
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

################################################################################
#Main program
[freqvec, ftaxislist, ftpulsexarray, ftpulsezarray, ftveclist, ftxdiparray,
 ftzdiparray,  namelist, pulsexarray, pulsezarray, veclist, xdiparray,
 zdiparray]=readpickledfiles()

savefigs=True

Earray=pulsezarray[:,0,0,0,:]
ftEarray=nDFFT(Earray,axislist=[-1], shiftaxes=True)
#for i in range(-3,4):
#    for j in range(-3,4):
#        if(mod(i-j,2)==0):

evlo=20
evhi=30
npr=1#number of probe photons

ylimstr="_ylo_"+str(evlo)+"eV_yhi_"+str(evhi)+"eV"
filenamebase="helium_nonhermitian_decomposition_npr_"+str(npr)+"Iir_3e11_Ixuv_1e10"
taarraylist=[]
filenamelist=[]
legendlist=[]
for Ntot in range(2,3):#range(-6,7,2):
    for i in range(-3,4):
        j=-(Ntot-i)
        nstr="_np_"+str(i)+"_nm_"+str(j)+"_ntot_"+str(Ntot)
        if(j in range(-3,4)):
            nlist=indxlist(ftveclist[1:-1],[npr,i,j])
            tmparray=ftzdiparray[:,nlist[0],nlist[1],nlist[2],:]
            taarray=transientabsorption(tmparray,ftEarray)
            taarraylist.append(taarray)
            legend=str(i)+", "+str(j)+", Ntot = "+str(i-j)
            legendlist.append(legend)
            filename=filenamebase+nstr+ylimstr+".png"
            filenamelist.append(filename)

#plot arrays after normalizing
xvec=veclist[0]
yvec=ftveclist[-1]
xarray, yarray=vecstoarrays(xvec,yvec)
maxval=array(list(map(lambda x: abs(x).max(), taarraylist))).max()
for i in range(len(taarraylist)):
    taarray=taarraylist[i]/maxval
    legend=legendlist[i]
    filename=filenamelist[i]
    retfig=imshowplot(xarray,yarray,taarray, xmultfactor=aut/1000, ymultfactor=Hrt,
                      symmetricrange=True, gamma=5, ylo=evlo, yhi=evhi,
                      legend=legend)
    if(savefigs):
        retfig.savefig(filename)
        print("saved figure "+filename)
plt.show()
