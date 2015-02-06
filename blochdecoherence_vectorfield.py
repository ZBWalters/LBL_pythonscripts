import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *
from scipy.special.orthogonal import p_roots
from scipy import *
from mayavi.mlab import *

#This program plots decaying coherence trajectories in the Bloch sphere
#representation for the avian compass. Here the z component decays as Gamma,
#while the x and y components decay as lambda



def xyzt((x0,y0,z0),t,Gm,Lm):
    x=x0*exp(-Lm*t)
    y=y0*exp(-Lm*t)
    z=z0*exp(-Gm*t)
    return (x,y,z)

def VxVyVz((x,y,z),Gm,Lm):
    Vx=-Lm*x
    Vy=-Lm*y
    Vz=-Lm*z
    return(Vx,Vy,Vz)

def x0y0z0(r,theta,phi):
    z0=r*cos(theta)
    x0=r*sin(theta)*cos(phi)
    y0=r*sin(theta)*sin(phi)
    return (x0,y0,z0)

def thetapts(npts):
    legarray=p_roots(npts)
    return arccos(legarray[0])

def phipts(npts):
    return arange(0.,2.*pi,2.*pi/npts)

def tpts(tmax,npts):
    return arange(0.,tmax,tmax/npts)

def trajectory((r0,theta0,phi0),Gm,Lm,tpts):
    (x0,y0,z0)=x0y0z0(r0,theta0,phi0)
    npts=len(tpts)
    xpts=zeros(npts)
    ypts=zeros(npts)
    zpts=zeros(npts)
    for i in range(npts):
        t=tpts[i]
        (x,y,z)=xyzt((x0,y0,z0),t,Gm,Lm)
        xpts[i]=x
        ypts[i]=y
        zpts[i]=z
    return (xpts,ypts,zpts)

def plotsphere(ax):
    u=np.linspace(0,2*pi,100)
    v=np.linspace(0,pi,100)
    x=np.outer(cos(u),sin(v))
    y=np.outer(sin(u),sin(v))
    z=np.outer(np.ones(np.size(u)),cos(v))
    ax.plot_surface(x,y,z,color="grey",shade="yes",alpha=0.5,linewidth=0.1,rstride=5,cstride=5)

#strength of coupling = d
d=1.
gamma=25.
nanglepts=9
ntpts=1000
nrpts=5

lmbda=(gamma-((1.+0j)*gamma**2-4.*d**2)**.5)/2.
deltat=2.*pi/d
tmax=10./gamma#1.*deltat
tarray=tpts(tmax,ntpts)

r0=1.
thetaarray=thetapts(nanglepts)
phiarray=phipts(nanglepts)
deltar=1./nrpts
rarray=arange(deltar,1.+deltar,deltar)

npts=len(thetaarray)*len(phiarray)*len(rarray)
xarray=zeros((nanglepts,nanglepts,nrpts))#zeros(npts)
yarray=zeros((nanglepts,nanglepts,nrpts))#zeros(npts)
zarray=zeros((nanglepts,nanglepts,nrpts))#zeros(npts)
Vxarray=zeros((nanglepts,nanglepts,nrpts))#zeros(npts)
Vyarray=zeros((nanglepts,nanglepts,nrpts))#zeros(npts)
Vzarray=zeros((nanglepts,nanglepts,nrpts))#zeros(npts)

for i in range(nanglepts):
    for j in range(nanglepts):
        for k in range(nrpts):
            theta=thetaarray[i]
            phi=phiarray[j]
            r=rarray[k]
            (x,y,z)=x0y0z0(r,theta,phi)
            (Vx,Vy,Vz)=VxVyVz((x,y,z),gamma,lmbda)
            xarray[i,j,k]=x
            yarray[i,j,k]=y
            zarray[i,j,k]=z
            Vxarray[i,j,k]=Vx
            Vyarray[i,j,k]=Vy
            Vzarray[i,j,k]=Vz

obj=quiver3d(xarray,yarray,zarray,Vxarray,Vyarray,Vzarray)
show()

#fig=plt.figure()
#ax=fig.gca(projection='3d')
#for theta in thetaarray:
#    for phi in phiarray:
#        (xpts,ypts,zpts)=trajectory((1.,theta,phi),gamma,lmbda,tarray)
#        ax.plot(xpts,ypts,zpts,color='b')
#plotsphere(ax)
#plt.show()

