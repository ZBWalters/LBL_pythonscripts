from numpy import *
from scipy import *
import sys
import os
sys.path.append(os.path.expanduser("~/pythonscripts/"))
from twoDplot import *
from numpy.fft import fft,ifft,fftfreq
from pylab import plot


def directionalcontinuity(anglelist,nstart,nfinish,step):
    retanglelist=anglelist*1.
    retanglelist[nstart]=mod(retanglelist[nstart],pi)
    lastangle=retanglelist[nstart]
    for i in range(nstart,nfinish,step):
        piint=round((retanglelist[i]-lastangle)/pi)
        retanglelist[i]=retanglelist[i]-pi*piint
        lastangle=retanglelist[i]
    return retanglelist


def imposecontinuity(anglelist):
    lhalf=int(len(anglelist)/2)
    tmplist=directionalcontinuity(anglelist,lhalf,len(anglelist),1)
    retlist=directionalcontinuity(tmplist,lhalf,-1,-1)
    return retlist
    #figure()
    #plot(yarray[ncol,:]*Hrt,log(abs(zarray[ncol,:])))
    #plot(yarray[ncol,:]*Hrt,angle(zarray[ncol,:])/yarray[ncol,:])

def impose_angle_continuity(inparray):
    (n1,n2)=shape(inparray)
    retarray=zeros((n1,n2))*0j
    for i in range(n1):
        retarray[i,:]=imposecontinuity(angle(inparray[i,:]))
    return retarray

def listderiv(inplist):
    n1=len(inplist)
    retlist=zeros(n1)*0j
    retlist[0]=inplist[1]-inplist[0]
    retlist[-1]=inplist[-1]-inplist[-2]
    for i in range(1,n1-1):
        retlist[i]=(inplist[i+1]-inplist[i-1])/2
    return retlist

def yderiv(yarray,inparray):
    dy=yarray[0,1]=yarray[0,0]
    (n1,n2)=shape(inparray)
    retarray=zeros((n1,n2))*0j
    for i in range(n1):
        retarray[i,:]=listderiv(inparray[i,:])
    retarray/=dy
    return retarray

def listavg(inplist):
    n1=len(inplist)
    retlist=zeros(n1)*0j
    retlist[0]=inplist[0]
    retlist[-1]=inplist[-1]
    for i in range(1,n1-1):
        retlist[i]=(inplist[i+1]+inplist[i-1])/2
    return retlist

def yavg(inparray):
    (n1,n2)=shape(inparray)
    retarray=zeros((n1,n2))*0j
    for i in range(n1):
        retarray[i,:]=listavg(inparray[i,:])
    return retarray

def gauss(x,width):
    return exp(-pow(abs((x)/width),2))*(1.+0j)/(sqrt(pi)*width)

def convolveyfun(xarray,yarray,zarray,fun,funargs):
    xvec=xarray[:,0]
    yvec=yarray[0,:]
    print("shapes\t"+str(shape(xarray))+"\t"+str(shape(yarray))+"\t"+str(shape(zarray)))
    mapfun=lambda x: fun(x,*funargs)
    yfunvals=array(list(map(mapfun,yvec)))
    yfprime=yfunvals*0j
    ylen=len(yfprime)
    for i in range(ylen):
        yfprime[i]=yfunvals[mod(i-ylen/2,ylen)]
    fftyfun=fft(yfprime)
    #print("fftyfun\t"+str(fftyfun))
#try using built in convolution
#    zfft=zarray*0j
#    (n1,n2)=shape(zarray)
#    for i in range(n1):
#        fftvec=convolve(zarray[i,:],yfunvals)
#        print("shape fftvec\t"+str(shape(fftvec)))
#        zfft[i,:]=fftvec

#try using fft convolution
    zfft=zarray*1.
    (n1,n2)=shape(zfft)
    for i in range(n1):
        fftvec=fft(zfft[i,:])
        for j in range(n2):
            zfft[i,j]=fftvec[j]*fftyfun[j]
    for i in range(n1):
        zfft[i,:]=ifft(zfft[i,:])
    return zfft


def timedelay(xarray,yarray,zarray):
    dzdy=yderiv(yarray,zarray)
    zavgarray=yavg(zarray)
    timedelayarray=imag(dzdy/zavgarray)#y deriv of phase =imaginary part
                                    #of d/dy log(z)
    return timedelayarray

def convtimedelay(xarray,yarray,zarray,width):
    timedelayarray=timedelay(xarray,yarray,zarray)
    return convolvepopweight(xarray,yarray,zarray,timedelayarray,width)

def convolvepopweight(xarray,yarray,zarray,valarray,width):
    zconv=convolveyfun(xarray,yarray,abs(zarray)**2,gauss,[width])
    valconv=convolveyfun(xarray,yarray,valarray*abs(zarray)**2,gauss,[width])
    return valconv/zconv

##########################################
#try taking phase derivative using fourier transform method
xarray,yarray,zarray=arraysfromfile("1/kIR_vs_freq.dat")
#xarray,yarray,zarray=arraysfromfile("1/phase_vs_freq.dat")

xvec=sorted(list(set(xarray.flatten())))
yvec=sorted(list(set(yarray.flatten())))

(n1,n2)=shape(zarray)


tdly=timedelay(xarray,yarray,zarray)
width=1./Hrt
ctdly=convtimedelay(xarray,yarray,zarray,width)

phasearray=impose_angle_continuity(zarray)
dphasedy=yderiv(yarray,phasearray)


