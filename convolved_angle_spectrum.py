import sys
import os
sys.path.append(os.path.expanduser("~/pythonscripts/"))
from twoDplot import *
from numpy.fft import fft,ifft
from scipy.signal import fftconvolve, convolve
from pylab import *

def gauss(x,width):
    return exp(-pow(abs((x)/width),2))*(1.+0j)

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

def fileconvolveplots(filename,cpwr=.5,width=1./Hrt):
    xarray,yarray,zarray=arraysfromfile(filename)
    zfft=convolveyfun(xarray,yarray,zarray,gauss,[width])
#    imshowplot_hsv(xarray,yarray*Hrt,zarray,colorpower=cpwr)
#    imshowplot_hsv(xarray,yarray*Hrt,zfft,colorpower=cpwr)
#    figure()
#    for ncol in range(12,24,2):
#        plot(yarray[ncol,:]*Hrt,log(abs(zfft[ncol,:])),label=str(ncol))
#    plt.legend()
#    figure()
#    for ncol in range(12,24,2):
#        plot(yarray[ncol,:]*Hrt,aut*angle(zfft[ncol,:])/yarray[ncol,:]+aut*pi/yarray[ncol,:])
    figure()
    for ncol in range(12,18,2):
        plot(yarray[ncol,:]*Hrt,imposecontinuity(angle(zarray[ncol,:])),label=str(ncol))
    legend()
    figure()
    for ncol in range(12,18,2):
        plot(yarray[ncol,:]*Hrt,angle(zarray[ncol,:]),label=str(ncol))
    legend()


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
