import numpy as np
import scipy as scp
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special.orthogonal import p_roots
from scipy import *

#This program plots decaying coherence trajectories in the Bloch sphere
#representation for the avian compass. Here the z component decays as Gamma,
#while the x and y components decay as lambda

ncirclepts=100

def xyzt((x0,y0,z0),t,Gm,Lm):
    x=x0*exp(-Lm*t)#(.5*exp(-Lm*t)+.5*exp(-conjugate(Lm)*t))
    y=y0*exp(-Lm*t)#(.5*exp(-Lm*t)+.5*exp(-conjugate(Lm)*t))#exp(-Lm*t)
    z=z0*exp(-Gm*t)
    return (x,y,z)

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
    u=np.linspace(0,2*pi,ncirclepts+1)
    v=np.linspace(0,pi,ncirclepts+1)
    x=np.outer(cos(u),sin(v))
    y=np.outer(sin(u),sin(v))
    z=np.outer(np.ones(np.size(u)),cos(v))
    nlines=10
    ax.plot_surface(x,y,z,color="grey",shade="yes",alpha=0.05,linewidth=0.1,rstride=ncirclepts/nlines,cstride=ncirclepts/nlines)
    #ax.plot_surface(x,y,z,color="grey",shade="yes",alpha=0.05,linewidth=0.1,rstride=10,cstride=10)
    plotequator(ax)
    plotequatorialplane(ax)
    plotaxes(ax)

def plotequator(ax):
    nthetapts=100.
    deltatheta=2.*pi/nthetapts
    thetapts=arange(0.,2.*pi+deltatheta,deltatheta)
    xpts=cos(thetapts)
    ypts=sin(thetapts)
    zpts=xpts*0.
    ax.plot(xpts,ypts,zpts,color='black',linewidth=1.)

def plotequatorialplane(ax):
    dr=.01
    dtheta=.05
    rpts=np.linspace(0,1,ncirclepts)
    thetapts=np.linspace(0,2.*pi,ncirclepts)
    xpts=np.outer(rpts,cos(thetapts))
    ypts=np.outer(rpts,sin(thetapts))
    (n1,n2)=shape(xpts)
    zpts=zeros((n1,n2))
    ax.plot_surface(xpts,ypts,zpts,color="grey",shade="yes",alpha=0.1,linewidth=0.,rstride=5,cstride=5)

def plotaxes(ax):
    xpts=linspace(-1.,1.,ncirclepts)
    ypts=xpts*0.
    zpts=xpts*0.
    ax.plot(xpts,ypts,zpts,color='black',linewidth=1.)
    ax.plot(ypts,xpts,zpts,color='black',linewidth=1.)
    ax.plot(ypts,zpts,xpts,color='black',linewidth=1.)

#strength of coupling = d
d=50*6e-4
gamma=10.
nthetapts=13
nphipts=2
ntpts=100

lmbda=(gamma-((1.+0j)*gamma**2-4.*d**2)**.5)/2.
deltat=2.*pi/d
tmax=.1/abs(lmbda)#10./gamma#1.*deltat
tarray=tpts(tmax,ntpts)

r0=1.
thetaarray=thetapts(nthetapts)
phiarray=phipts(nphipts)

fig=plt.figure()
ax=fig.gca(projection='3d',azim=90,elev=15)
ax._axis3don=False
for theta in thetaarray:
    for phi in phiarray:
        (xpts,ypts,zpts)=trajectory((1.,theta,phi),gamma,lmbda,tarray)
        ax.plot(xpts,ypts,zpts,linewidth=2.)#,color='b')
plotsphere(ax)
#plotequator(ax)
txtsize='x-large'
ax.text(0,0,1.2,r'$|\uparrow \downarrow >$',size=txtsize)
ax.text(0,0,-1.2,r'$|\downarrow \uparrow >$',size=txtsize)
ax.text(1.2,0,0,r'$|s>$',size=txtsize,color='black')
ax.text(-1.2,0,0,r'$|t>$',size=txtsize,color='black')
ax.w_xaxis.set_ticks([0.])
ax.w_yaxis.set_ticks([0.])
ax.w_zaxis.set_ticks([0.])
#plt.savefig('fig.png',bbox_inches='tight',pad_inches=0.)
plt.savefig('fig.png',dpi=1000,bbox_inches='tight',pad_inches=0.)
#plt.savefig('fig.eps',bbox_inches='tight',pad_inches=0.)
plt.show()

