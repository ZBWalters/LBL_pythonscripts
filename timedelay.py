from twoDplot import *
from numpy.fft import *
#use the tools developed in twoDplot.py to plot figures from optical streaking

def dpdElogz(inparray,warray):
    retarray=zeros(shape(inparray))*0j
    (n0,n1)=shape(inparray)
    for i,j in product(range(n0),range(1,n1-1)):
        #retarray[i,j]=(log(inparray[i,j+1])-log(inparray[i,j-1]))/(warray[i,j+1]-warray[i,j-1])
        a=inparray[i,j+1]
        b=inparray[i,j-1]
        dw=warray[i,j+1]-warray[i,j-1]
        retarray[i,j]=(a-b)/(a+b)*2/dw
    return retarray
        
def fdlogderiv(inparray,warray):
    retarray=zeros(shape(inparray))*0j
    (n0,n1)=shape(inparray)
    for i,j in product(range(n0),range(1,n1-1)):
        maga=abs(inparray[i,j+1])
        magb=abs(inparray[i,j-1])
        arga=angle(inparray[i,j+1])
        argb=angle(inparray[i,j-1])
        r=abs(inparray[i,j])
        theta=angle(inparray[i,j])
        dw=warray[i,j+1]-warray[i,j-1]
        drdw=(maga-magb)/dw
        dthetadw=mod(arga-argb,2*pi)/dw
        retarray[i,j]=((drdw*cos(theta)-r*sin(theta)*dthetadw)+
                       1j*(drdw*sin(theta)+r*cos(theta)*dthetadw))/inparray[i,j]
        print("i,j,retarray\t"+str(i)+"\t"+str(j)+"\t"+str(retarray[i,j]))
    return retarray

def dtheta(theta1,theta2):
    dangle=mod(theta1-theta2,2*pi)
    if(dangle>pi):
        dangle-=2*pi
    return dangle

def logderiv(inparray,warray):
    retarray=zeros(shape(inparray))*0j
    (n0,n1)=shape(inparray)
    for i,j in product(range(n0),range(1,n1-1)):
        a=log(inparray[i,j+1])
        b=log(inparray[i,j-1])
        dw=warray[i,j+1]-warray[i,j-1]
        retarray[i,j]=real(a)-real(b)+1j*dtheta(imag(a),imag(b))
        retarray[i,j]/=dw
    return retarray


def nDFFT(canglediparray,axislist, shiftaxes=False, inv=False):
    if(inv):
        fftdiparray=ifftn(canglediparray,axes=axislist)
    else:
        fftdiparray=fftn(canglediparray,axes=axislist)
    if(shiftaxes):
        if(inv):
            fftdiparray=ifftshift(fftdiparray,axes=axislist)
        else:
            fftdiparray=fftshift(fftdiparray,axes=axislist)
    return fftdiparray

def removedrivingfreq(narray,tarray,diparray,wosc=.0565):
    retarray=copy(diparray)
    retarray*=exp(-1j*wosc*narray*tarray)
    return retarray

def maskarray(inparray,xwidth=1,ywidth=1):
    (n0,n1)=shape(inparray)
    retarray=zeros((n0,n1))*0j
    m0=ceil(n0/2)
    m1=ceil(n1/2)
    for i,j in product(range(n0),range(n1)):
        retarray[i,j]=exp(-pow((i-m0)/xwidth,2))*exp(-pow((j-m1)/ywidth,2))
    retarray=nDFFT(retarray,axislist=[1],shiftaxes=True,inv=True)
    retarray=nDFFT(retarray,axislist=[0],shiftaxes=True)
    return retarray



phiarray,warray,pwarray=arraysfromfile("1/phase_vs_freq.dat")
narray,tarray,diparray=arraysfromfile("1/kIR_vs_t.dat")



newdiparray=removedrivingfreq(narray,tarray,diparray)*maskarray(diparray,ywidth=.3,
                                                                xwidth=.3)
ftnewdiparray=nDFFT(newdiparray,axislist=[1],shiftaxes=True)
ftnewdiparray=nDFFT(ftnewdiparray,axislist=[0],shiftaxes=True,inv=True)


small=0#1.e-2*abs(ftnewdiparray).max()
dpwdElogzarray=logderiv(ftnewdiparray+small,warray)#dpdElogz(pwarray,warray)
pltindx=20
plt.plot(warray[pltindx,:],imag(dpwdElogzarray)[pltindx,:])
imshowplot_hsv(phiarray,warray,imag(dpwdElogzarray))

