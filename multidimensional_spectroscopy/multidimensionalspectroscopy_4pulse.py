from numpy import *
from scipy import *
from pylab import *
from scipy.integrate import ode, complex_ode
from numpy.fft import fft2

#define constants
Eex=1.
Esplit=.25
Eod=.05

E0=0.
E1=E0+Eex-Esplit/2.
E2=E0+Eex+Esplit/2.

gamma=0.#.1
gamma1=gamma
gamma2=gamma

d=.1
d01=d
d02=d

w1=Eex
w2=Eex
w3=Eex
w4=Eex

F0=.1
F1=.1
F2=.1
F3=.1
F4=.1

ncycle=1.
tau=ncycle*pi/Eex
tau1=tau
tau2=tau
tau3=tau
tau4=tau


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

pmat=zeros((3,3))*1j
pmat[0,0]=1.

def gauss(t,tcenter,tau):
    return exp(-pow((t-tcenter)/tau,2.))/(tau*sqrt(pi))

def subpulse(w,t,tcenter,tau):
    return cos(w*t)*gauss(t,tcenter,tau)

def pulse(t,t1,t2,t3,t4):
    return F1*subpulse(w1,t,t1,tau1)+F2*subpulse(w2,t,t2,tau2)+F3*subpulse(w3,t,t3,tau3)+F4*subpulse(w4,t,t4,tau4)

def comm(A,B):
    return dot(A,B)-dot(B,A)

def dpvecdt_H(t,pvec,t1,t2,t3,t4):
    tmppmat=reshape(pvec,(3,3))
    tmpHmat=Hmat+pulse(t,t1,t2,t3,t4)*Dmat
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

def dpvecdt(t,pvec,t1,t2,t3,t4):
    #print "pvec",reshape(pvec,(3,3))
    dpvecdt=dpvecdt_H(t,pvec,t1,t2,t3,t4)+dpvecdt_G(pvec)
    #print "dpvecdt",reshape(dpvecdt,(3,3))
    return dpvecdt

def pexpec(pvec):
    tmppmat=reshape(pvec,(3,3))
    return trace(dot(Dmat,tmppmat))
        

#########
def pdeint(pde,t0,tf,dt):
    tnext=t0
    y=pde.y
    print "pdeint t0,tf,dt",t0,tf,dt
    while(tnext<tf):
        tnext=min(tnext+dt,tf)
        print "tnext",tnext
        y=pde.integrate(tnext)
    return y
    

#def E3dotP3(dt1,dt2,step,dt3array):
#    ##integrate pmat * calculate Trace(Dmat.pmat)(t)*subpulse(w3,t,t3,tau3)
#    t0=0.#-4.*tau1
#    t1=0.
#    t2=dt1
#    t3=t2+dt2
#    evaltimes=dt1+dt2+dt3array
#    PdotEarray=evaltimes*0j
#
#    y0=reshape(pmat,(9)) # initial pvec
#    pde=ode(dpvecdt).set_integrator('zvode')
#
#    pde.set_initial_value(y0,t0)
#    pde.set_f_params(t1,t2,t3)
#
#    tstart=t0
#    for i in range(len(evaltimes)):
#        tnext=evaltimes[i]
#        y=pdeint(pde,tstart,tnext,step)
#        tstart=tnext
#        print "y",tnext, y
#        pval=pexpec(y)
#        PdotEarray[i]=pval*subpulse(w3,tnext,t3,tau3)
#    return [evaltimes,PdotEarray]
 

def excitedpops(dt1,dt2,dt3,step):
    ##integrate pmat * calculate Trace(Dmat.pmat)(t)*subpulse(w3,t,t3,tau3)
    t0=-4.*tau1
    t1=0.
    t2=dt1
    t3=t2+dt2
    t4=t3+dt3
    tstop=t4+4.*tau4
    
    y0=reshape(pmat,(9)) # initial pvec
    #pde=ode(dpvecdt).set_integrator('zvode',method='adams',min_step=1.e-4,max_step=10.,nsteps=1e6)
    pde=ode(dpvecdt).set_integrator('zvode',method='adams',nsteps=1e6,rtol=1.e-4)


    #pde=complex_ode(dpvecdt).set_integrator('dopri5')

    pde.set_initial_value(y0,t0)
    pde.set_f_params(t1,t2,t3,t4)
    
    tmppvec=pde.integrate(tstop)
    #print "calling pdeint step", step
    #pdeint(pde,t0,tstop,step)
    #tmppvec=pde.y
    tmppmat=reshape(tmppvec,(3,3))
    return [tmppmat[0,0],tmppmat[1,1],tmppmat[2,2]]
   
    

############################################################
#####Make Plots

#deltat arrays
dtmin=0.
dtmax=20.*pi/Esplit
ndt=20
dtarray=arange(dtmin,dtmax,(dtmax-dtmin)/ndt)
tstep=.01


#1st index of EParray is dt2, 2nd is dt1, 3rd is dt3
pop0array=zeros((ndt,ndt,ndt))*0j
pop1array=zeros((ndt,ndt,ndt))*0j
pop2array=zeros((ndt,ndt,ndt))*0j

#simple test
#dt1=dtarray[10]
#dt2=dtarray[10]
#dt3=dtarray[10]
#tstep=.01
#[pop0,pop1,pop2]=excitedpops(dt1,dt2,dt3,tstep)
#print "pop0,pop1,pop2",pop0,pop1,pop2

for i in range(ndt/2,ndt/2+1):
    dt2=dtarray[i]
    for j in range(ndt):
        dt1=dtarray[j]
        for k in range(ndt):
            dt3=dtarray[k]
            [pop0,pop1,pop2]=excitedpops(dt1,dt2,dt3,tstep)
            pop0array[i,j,k]=pop0
            pop1array[i,j,k]=pop1
            pop2array[i,j,k]=pop2

##take 2d fourier transforms
pop0fftarray=fft2(pop0array)
pop1fftarray=fft2(pop1array)
pop2fftarray=fft2(pop2array)



pop0fftarray=fftshift(pop0fftarray,(1,2))
pop1fftarray=fftshift(pop1fftarray,(1,2))
pop2fftarray=fftshift(pop2fftarray,(1,2))
#plot figure
imshow(log(abs(pop1fftarray[ndt/2,:,:])))
#imshow(log(abs(EPfftarray[10,:,:])))
show()
