from numpy import *
from scipy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import glob

#######################
#utility functions
def uniquevals(inparray):
    retarray=[]
    for val in inparray:
        if val not in retarray:
            retarray.append(val)
    #print retarray
    return retarray

def pltarrays(xarray,yarray,zarray):
    #converts 1D arrays to 2D arrays which
    #can be used for plotting
    xvals=uniquevals(xarray)
    nx=len(xvals)
    yvals=uniquevals(yarray)
    ny=len(yvals)
    
    #print "xvals",xvals
    #print "yvals",yvals

    pltx=xarray.reshape((nx,ny))
    plty=yarray.reshape((nx,ny))

    #print "pltx",pltx

    #print "plty",plty

    pltz=zarray.reshape((nx,ny))
    return [pltx,plty,pltz]

#############################
def scatterplotarray(pltarray,xcol,ycol,zcol):
    return scatterplot(pltarray[xcol,:],pltarray[ycol,:],pltarray[zcol,:])

def scatterplot(xarray,yarray,zarray):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(xarray,yarray,zarray,cmap=plt.cm.Spectral)
    fig.colorbar()
    return fig

def trisurfplot(xarray,yarray,zarray,plotpoints='false',sinp=5.):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_trisurf(xarray,yarray,zarray,cmap=plt.cm.Spectral)
    if(plotpoints):
        ax.scatter(xarray,yarray,zarray,s=sinp,cmap=plt.cm.Spectral)

    fig.colorbar()
    return fig

def pcolorplot(xarray,yarray,zarray,inpnbins=15):
    [pltx,plty,pltz]=pltarrays(xarray,yarray,zarray)

    print "shapes ",shape(pltx),shape(plty),shape(pltz)

    levels = MaxNLocator(nbins=inpnbins).tick_values(pltz.min(), pltz.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('Spectral')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig=plt.figure()
    #ax=fig.add_subplot(111,projection='3d')
    plt.pcolormesh(pltx,plty,pltz,cmap=cmap)#,norm=norm)
    #ax.axis([pltx.min(),pltx.max(),plty.min(),plty.max()])
    plt.colorbar()
    return fig

def imshowplot(xarray,yarray,zarray):
    [pltx,plty,pltz]=pltarrays(xarray,yarray,zarray)
    fig=plt.figure()
    plt.imshow(pltz,vmin=zarray.min(),vmax=zarray.max(),extent=[xarray.min(),xarray.max(),yarray.min(),yarray.max()],origin='lower')
    plt.colorbar()
    return fig
