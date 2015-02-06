from numpy import *
from scipy import *
from scipy.optimize import brentq
from pylab import *


def FofI(intensity):#converts intensity in watts per square centimeter
                    #to atomic units of electric field (see atomic
                    #units handout)
    au_per_Wpercmsq=2.8e-17
    return sqrt(intensity*au_per_Wpercmsq)

def A(t):#vector potential
    return (F0/w)*sin(w*t)

def E(t):#electric field
    return -F*cos(w*t)

def pconjg(p,t):#conjugate momentum
    return p-q*A(t)

def pc_tion(tion):#conjugate momentum for trajectory ionizing with
                  #zero initial velocity
    return pconjg(0,tion)

def KE(p,t):#kinetic energy
    return pow(pconjg(p,t),2)/2

def v(pconjg,t):#instantaneous velocity
    return pconjg+q*A(t)

def trajectory_v(tion,t):
    pc=pconjg(0,tion)
    return v(pc,t)

def trajectory_x(tion,v0,t):
    return pconjg(v0,tion)*(t-tion)-q*F0/(pow(w,2))*(cos(w*t)-cos(w*tion))

def treturn(tion,v0):
    xt=lambda t: trajectory_x(tion,v0,t)
    t1=tion+.1
    t2=tion+2*pi/w
    tret=0.
    if(xt(t1)*xt(t2)>0):
        tret=0.
    else:
        tret=brentq(xt,t1,t2)
    return tret

def KEreturn(tion,v0):
    tret=treturn(tion,v0)
    vret=trajectory_v(tion,tret)
    return KE(vret,tret)

#################################
#Main program
I0=1e14#watts per square centimeter


F0=FofI(I0)#.1
w=.0565*2
q=-1

Hrt=27.21


#plot properties of the returning trajectories
dt=.1
tionarray=arange((pi/10)/w,(pi/2)/w,dt)

#return times
tretarray=tionarray*0.
for i in range(len(tionarray)):
    tretarray[i]=treturn(tionarray[i],0.)

fig1=figure()
plot(tionarray*w,tretarray*w)


#return velocities
vretarray=tretarray*0.
for i in range(len(tretarray)):
    if(tretarray[i]!=0.):
        vretarray[i]=trajectory_v(tionarray[i],tretarray[i])

fig2=figure()
plot(tionarray*w,vretarray)


#return KEs
KEretarray=tretarray*0.
for i in range(len(tretarray)):
    if(tretarray[i]!=0.):
        KEretarray[i]=KEreturn(tionarray[i],0.)

fig3=figure()
plot(tionarray*w,KEretarray*Hrt)
legend("KE vs tion")

fig4=figure()
plot(tretarray*w,KEretarray*Hrt)
legend("KE vs tion")



                

