from numpy import *
from scipy import *
from pylab import *
from scipy.integrate import ode
from numpy.fft import fft2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#define constants
Eex=1.
Esplit=.1
Eod=0.1

E0=0.
E1=E0+Eex-Esplit/2.
E2=E0+Eex+Esplit/2.

gamma=0.#.1
gamma1=gamma
gamma2=gamma

F1=.1#intensity of 1st pulse
F2=.1
F3=.1


d=.1
d01=d
d02=2.*d
d12=0.

w1=Eex+Esplit/2.
w2=Eex+Esplit/2.
w3=Eex+Esplit/2.

ncycle=2.
tau=ncycle*pi/w1
tau1=ncycle*pi/w1
tau2=ncycle*pi/w2
tau3=ncycle*pi/w3


#set up H, D, p mats
Hmat=zeros((3,3))*1j
Hmat[0,0]=E0
Hmat[1,1]=E1
Hmat[2,2]=E2
Hmat[1,2]=Eod
Hmat[2,1]=Eod

Dmat=zeros((3,3))*1j
Dmat[0,1]=d01
Dmat[1,0]=d01
Dmat[0,2]=d02
Dmat[2,0]=d02
Dmat[1,2]=d12
Dmat[2,1]=d12

pmat=zeros((3,3))*1j
pmat[0,0]=1.

def gauss(t,tcenter,tau):
    return exp(-pow((t-tcenter)/tau,2.))/(tau*sqrt(pi))

def subpulse(w,t,tcenter,tau):
    return cos(w*t)*gauss(t,tcenter,tau)

def pulse(t,t1,t2,t3):
    return F1*subpulse(w1,t,t1,tau1)+F2*subpulse(w2,t,t2,tau2)+F3*subpulse(w3,t,t3,tau3)

def comm(A,B):
    return dot(A,B)-dot(B,A)

def dpvecdt_H(t,pvec,t1,t2,t3):
    tmppmat=reshape(pvec,(3,3))
    tmpHmat=Hmat+pulse(t,t1,t2,t3)*Dmat
    tmppdot=1j*comm(tmpHmat,tmppmat)
    return reshape(tmppdot,(9))

def dpvecdt_G(pvec):
    tmppmat=reshape(pvec,(3,3))
    tmppdot=zeros((3,3))*1j
    for i in range(3):
        tmppdot[1,i]-=gamma1*tmppmat[1,i]/2.
        tmppdot[i,1]-=gamma1*tmppmat[i,1]/2.
        tmppdot[2,i]-=gamma2*tmppmat[2,i]/2.
        tmppdot[i,2]-=gamma2*tmppmat[i,2]/2.
    return reshape(tmppdot,(9))

def dpvecdt(t,pvec,t1,t2,t3):
    return dpvecdt_H(t,pvec,t1,t2,t3)#+dpvecdt_G(pvec)

def pexpec(pvec):
    tmppmat=reshape(pvec,(3,3))
    return trace(dot(Dmat,tmppmat))
        

#########
def pdeint(pde,t0,tf,dt):
    tnext=t0
    y=pde.y
    while(tnext<tf):
        tnext=min(tnext+dt,tf)
        y=pde.integrate(tnext)
    return y
    

def E3dotP3(dt1,dt2,dt3array):
    ##integrate pmat * calculate Trace(Dmat.pmat)(t)*subpulse(w3,t,t3,tau3)
    t0=-4.*tau1
    t1=0.
    t2=dt1
    t3=t2+dt2
    t4=t3+5.*tau3#test: look at P3 after pulse is over
    evaltimes=dt1+dt2+dt3array+t4
    PdotEarray=evaltimes*0j

    pmat=zeros((3,3))
    pmat[0,0]=1.
    y0=reshape(pmat,(9)) # initial pvec
    pde=ode(dpvecdt).set_integrator('zvode',method='adams',rtol=1.e-4,nsteps=1e6,max_step=1.)

    pde.set_initial_value(y0,t0)
    pde.set_f_params(t1,t2,t3)

    tstart=t0
    for i in range(len(evaltimes)):
        tnext=evaltimes[i]
        #y=pdeint(pde,tstart,tnext,step)
        y=pde.integrate(tnext)
        pde.set_initial_value(y,tnext)
        tstart=tnext
        #print "y",tnext, y
        pval=pexpec(y)
        #PdotEarray[i]=pow(abs(pval+subpulse(w3,tnext,t3,tau3)*F3),2)
        PdotEarray[i]=pval
    return [evaltimes,PdotEarray]
    

def checkintegration(dt1,dt2,dt3array):
    t0=-4.*tau1
    t1=0.
    t2=dt1
    t3=t2+dt2+dt3array[-1]
    
    nevaltimes=1000
    evaltimes=arange(t0,t3,(t3-t1)/(nevaltimes*1.))
    print "evaltimes",evaltimes
    pop1=evaltimes*0j
    pop2=evaltimes*0j
    pop3=evaltimes*0j

    pmat=zeros((3,3))
    pmat[0,0]=1.
    y0=reshape(pmat,(9)) # initial pvec
    pde=ode(dpvecdt).set_integrator('zvode',method='adams',rtol=1.e-4,nsteps=1e6,max_step=1.)

    pde.set_initial_value(y0,t0)
    pde.set_f_params(t1,t2,t3)

    tstart=t0
    yarray=zeros((10,nevaltimes),dtype='complex')
    print "yarray",yarray
    for i in range(nevaltimes):
        print "i",i
        tnext=evaltimes[i]
        #y=pdeint(pde,tstart,tnext,step)
        pde.integrate(tnext)
        y=pde.y
        pde.set_initial_value(y,tnext)
        yarray[0,i]=evaltimes[i]
        yarray[1:,i]=y[:]
    print yarray[0,:]
    print yarray[1,:]
    for j in [2,3]:
        plot(yarray[0,:],abs(yarray[j,:]))
    show()
        
        

     
    
    
    

############################################################
#####Make Plots

#deltat arrays
ndt=40

dtinterval=5*pi/Esplit
dtmin=0.
dtmax=dtinterval
dtarray=arange(dtmin,dtmax,(dtmax-dtmin)/ndt)

dt3interval=4.*tau3
dt3min=-dtinterval/2.
dt3max=dtinterval/2.
dt3array=arange(dt3min,dt3max,(dt3max-dt3min)/ndt)
tstep=.01


#check integration
#dt1=dtarray[ndt/2]
#dt2=dtarray[-1]
#checkintegration(dt1,dt2,dt3array)



##ndt=len(dtarray)
#1st index of EParray is dt2, 2nd is dt1, 3rd is dt3
EParray=zeros((ndt,ndt,ndt))*0j
for i in range(ndt/2,ndt/2+1):
    for j in range(ndt):
       [t3array,PdotEarray]=E3dotP3(dtarray[j],dtarray[i],dt3array)
       for k in range(ndt):
           EParray[i,j,k]=PdotEarray[k]

##take 2d fourier transforms
EPfftarray=fft2(EParray)


#fourier transform frequencies
fftfreq1=sort(fftfreq(len(dtarray),dtarray[1]-dtarray[0]))*pi
fftfreq3=sort(fftfreq(len(dt3array),dt3array[1]-dt3array[0]))*pi
xarray=zeros((ndt,ndt))
yarray=zeros((ndt,ndt))
for i in range(ndt):
    for j in range(ndt):
        xarray[i,j]=fftfreq1[i]
        yarray[i,j]=fftfreq3[j]


#subtract away background
subpulse3array=pow(F3,2)*pow(abs(subpulse(w3,dtarray,0,tau3)),2)
fftsubpulse3array=fft(subpulse3array)
#fftsubpulse3array=fftshift(fftsubpulse3array)

#for i in range(ndt):
#    for j in range(ndt):
#        for k in range(ndt):
#            EPfftarray[i,j,k]-=fftsubpulse3array[k]
#
EPfftarray=fftshift(EPfftarray,(1,2))

#plot figure
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
#ax.scatter(xarray,yarray,real(EPfftarray[ndt/2,:,:]),cmap=cm.jet)
#ax.plot_wireframe(xarray,yarray,real(EPfftarray[ndt/2,:,:]),cmap=cm.jet)
#surf=ax.plot_surface(xarray,yarray,real(EPfftarray[ndt/2,:,:]),cmap=cm.jet,rstride=1,cstride=1)

surf=ax.plot_trisurf(xarray[:,:].flatten(),yarray[:,:].flatten(),(real(EPfftarray[ndt/2,:,:])).flatten(),cmap=cm.jet,linewidth=0.)
#surf=ax.plot_trisurf(xarray[ndt/2:,ndt/2:].flatten(),yarray[ndt/2:,ndt/2:].flatten(),abs(EPfftarray[ndt/2,ndt/2:,ndt/2:]).flatten(),cmap=cm.jet,linewidth=0.)
fig.colorbar(surf,cmap=cm.jet)
show()

##plot figure
##imshow(log(abs(EPfftarray[ndt/2,:,:]/F3)))
#imshow(real(EPfftarray[ndt/2,:,:]))
#
##imshow(log(abs(EPfftarray[10,:,:])))
#show()
