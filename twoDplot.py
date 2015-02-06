from numpy import *
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC
from scipy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap,hsv_to_rgb
from matplotlib.ticker import MaxNLocator
from matplotlib.image import NonUniformImage
import colorsys
from colorsys import hls_to_rgb
from colorsys import hsv_to_rgb as hsv2rgb
from itertools import product
from skimage.color import lab2rgb,luv2rgb
from matplotlib._cm import cubehelix
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot, make_axes_locatable

import os
import sys
import glob

#####useful constants
Hrt=27.21#Hartree in eV
aut=24.2
wavenum=8065.#wavenumbers per eV

def datfromfile(filename):
    #read data from a file
    loadarray=loadtxt(filename)
    xdat=loadarray[:,0]
    ydat=loadarray[:,1]
    zdat=loadarray[:,2]+1j*loadarray[:,3]
    return [xdat,ydat,zdat]

def uniquevals(inplist):
    #unique elements of a list (not guaranteed to preserve original
    #list's order)
    retlist=[]
    for elt in inplist:
        if (not (elt in retlist)):
            retlist.append(elt)
    retlist.sort()
    return retlist

def round_sig(x,sig=2):
    print("round_sig\t"+str(x))
    return sign(x)*round(abs(x),sig-int(floor(log10(abs(x))))-1)

def str_scientific(x,sig=1):
    if(x==0):
        retstr="0"
    else:
        power=int(floor(log(abs(x))/log(10))) 
        coeff=x/pow(10,power)
        coeff=round(coeff,sig)
        retstr= str(coeff)+"$\\times10^{"+str(power)+"}$"
    return retstr

def dattoarrays(xdat,ydat,zdat):
    #convert one dimensional dat arrays to two dimensional arrays
    n=len(uniquevals(xdat))
    m=len(uniquevals(ydat))
    xarray=reshape(xdat,(n,m))
    yarray=reshape(ydat,(n,m))
    yvec=yarray[0,:]
    zarray=reshape(zdat,(n,m))
    return [xarray,yarray,zarray]

def arraystodat(xarray,yarray,zarray):
    return xarray.flatten(),yarray.flatten(),zarray.flatten()

def arraysfromfile(filename):
    xdat,ydat,zdat=datfromfile(filename)
    xarray,yarray,zarray=dattoarrays(xdat,ydat,zdat)
    return xarray,yarray,zarray

def expandvec(xvec):
    #expand a coordinate vector such that pcolor will put points in
    #the right spots
    n=len(xvec)
    retvec=zeros(n+1)
    dx=xvec[1]-xvec[0]
    for i in range(n):
        retvec[i]=xvec[i]-dx/2
    retvec[n]=xvec[n-1]+dx/2
    return retvec

def maskarray(xarray, yarray, zarray, xlo=None, xhi=None, ylo=None, yhi=None,
              xmultfactor=1., ymultfactor=1.):
    if(xlo==None):
        xloval=-Infinity
    else:
        xloval=xlo
    if(xhi==None):
        xhival=Infinity
    else:
        xhival=xhi
    if(ylo==None):
        yloval=-Infinity
    else:
        yloval=ylo
    if(yhi==None):
        yhival=Infinity
    else:
        yhival=yhi

    retarray=copy(zarray)

    (n1,n2)=shape(xarray)
    #print("xloval,xhival,yloval,yhival\t"+str([xloval,xhival,yloval,yhival]))
    print("shape xarray\t"+str(shape(xarray)))
    print("shape yarray\t"+str(shape(yarray)))
    print("shape zarray\t"+str(shape(zarray)))
    for i,j in product(range(n1),range(n2)):
        xval=xarray[i,j]*xmultfactor
        yval=yarray[i,j]*ymultfactor
        #print("xval,yval\t"+str([xval,yval]))
        if(not((xval>=xloval) and (xval<=xhival) and (yval>= yloval) and (yval<=yhival))):
            retarray[i,j]=0.
            #zval=zrgbarray[j,i,1]
            #print("zval,zmin,zmax\t"+str([zval,zmin,zmax]))
            #if(zval<zmin):
            #    zmin=zval
            #if(zval>zmax):
            #    zmax=zval
    return retarray
    

##############################
def pcolorplot(xdat,ydat,zdat,absplot=False,colorpower=1,inpcmapname='hls'):
    #use pcolor() to plot xday,ydat,zdat

    #absplot controls whether the magnitude of zdat or its value
    #(which could be negative) is plotted

    #colorpower sets up a nonlinear colorbar scale, so that small
    #values can be accentuated.  Use colorpower <1, always
    fig=plt.figure()
    [xarray,yarray,zarray]=dattoarrays(xdat,ydat,zdat)
    xvec=expandvec(xarray[:,0])
    yvec=expandvec(yarray[0,:])
    if(absplot):
        cdict=abspowercmap(colorpower,inpcmapname)
        cmtmp=LinearSegmentedColormap('from_list',cdict)
        plt.pcolormesh(xvec,yvec,transpose(abs(zarray)),cmap=cmtmp)#cmap="Blues")
        plt.colorbar()
        zmax=abs(zarray).max()
        plt.clim(0,zmax)
    else:
        cdict=powercmap(colorpower,inpcmapname)
        cmtmp=LinearSegmentedColormap('from_list',cdict)
        plt.pcolormesh(xvec,yvec,transpose(zarray),cmap=cmtmp)#cmap="Spectral")
        plt.colorbar()
        zmax=abs(zarray).max()
        plt.clim(-zmax,zmax)
    return fig

def pcolorplotfile(filename,absplot=False,colorpower=1):
    #plot the data contained in a file using pcolor()
    [xdat,ydat,zdat]=datfromfile(filename)
    fig=pcolorplot(xdat,ydat,zdat,absplot,colorpower)
    return fig

def interpplot(xdat,ydat,zdat,absplot=False,colorpower=1,inpcmapname='hls',\
               legend="",xlabel="",ylabel="",xmultfactor=1.,ymultfactor=1.,\
               xlo=None,xhi=None,ylo=None,yhi=None,zmax=None):
    #plot xdat, ydat, zdat using NonUniformImage (allows interpolation)

    #absplot controls whether the magnitude of zdat or its value
    #(which could be negative) is plotted

    #colorpower sets up a nonlinear colorbar scale, so that small
    #values can be accentuated.  Use colorpower <1, always

    #legend is a string which can be used to label the plot
    
    fig=plt.figure()
    interp='bilinear'#'bilinear'#'nearest'
    [xarray,yarray,zarray]=dattoarrays(xdat,ydat,zdat)
    xvec=xmultfactor*xarray[:,0]
    xmin=xvec.min()
    if(xlo!=None):
        xmin=xlo
    xmax=xvec.max()
    if(xhi!=None):
        xmax=xhi
    yvec=ymultfactor*yarray[0,:]
    ymin=yvec.min()
    if(ylo!=None):
        ymin=ylo
    ymax=yvec.max()
    if(yhi!=None):
        ymax=yhi
    ax=fig.add_subplot(111)
    if(absplot):
        cdict=abspowercmap(colorpower,inpcmapname='hot')
        cmtmp=LinearSegmentedColormap('from_list',cdict)
    else:
        #cmtmp="RdBu"#"Spectral"#"RdBu"
        cdict=powercmap(colorpower,inpcmapname)
        cmtmp=LinearSegmentedColormap('from_list',cdict)
    im=NonUniformImage(ax,interpolation=interp,extent=(xmin,xmax,ymin,ymax),cmap=cmtmp)
    print("shapes ",str(shape(xvec))+"\t"+str(shape(yvec))+"\t"+str(shape(zarray))+"\n")
    if(absplot):
        im.set_data(xvec,yvec,transpose(abs(zarray)))
    else:
        im.set_data(xvec,yvec,transpose(real(zarray)))
    ax.images.append(im)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print("xmin,xmax,ymin,ymax "+str(xmin)+"\t"+str(xmax)+"\t"+str(ymin)+"\t"+str(ymax)+"\n")
    if(zmax==None):
        zmax=abs(real(zarray)).max()#sort(abs(zarray).flatten())[-int(len(zarray)*.1)]
    print("zmax\t"+str(zmax))
    if(absplot):
        im.set_clim(0,zmax)
    else:
        im.set_clim(-zmax,zmax)
    plt.colorbar(im)
    plt.title(legend)
    return fig

def interpplotfile(filename,absplot=False,colorpower=1,inpcmapname='hls',legend="",xlabel="",ylabel="",xmultfactor=1.,ymultfactor=1.,xlo=None,xhi=None,ylo=None,yhi=None):
    #plot the data contained in a file using interpplot()
    [xdat,ydat,zdat]=datfromfile(filename)
    fig=interpplot(xdat,ydat,zdat,absplot,colorpower,inpcmapname,legend,xlabel,ylabel,\
                   xmultfactor,ymultfactor,xlo,xhi,ylo,yhi)
    #fig.savefig(filename[:-4]+".png")
    return fig


def contourplot(xarray,yarray,zarray, colorpower="log",
                logrange=6,legend="",xlabel="",ylabel="",
                theme='hsv',xmultfactor=1.,ymultfactor=1.,
                xlo=None,xhi=None,ylo=None,yhi=None,
                inpfontsize=15,absplot=False,absmax=None,
                ncontours=5, contouralpha=1.,densityalpha=1.,
                contourcolors='w', contourcmap=None, contourcmapname='cubehelix_r',
                contourlinewidths=2, backgroundcolor="white"):
    
    if(contourcmap==None):
        contourcmap=plt.cm.get_cmap(contourcmapname)
    #print("showcontours?\t"+str(showcontours))
    interp="bilinear"#"none"#"bilinear"#'none'#"gaussian"#"bilinear"#"none"#"bilinear"
    print("xmultfactor, ymultfactor\t"+str(xmultfactor)+"\t"+str(ymultfactor))
    #[pltx,plty,pltz]=pltarrays(xarray,yarray,zarray)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.axes[0].set_axis_bgcolor(backgroundcolor)
    maskzarray=maskarray(xarray,yarray,zarray,xlo,xhi,ylo,yhi,
                         xmultfactor,ymultfactor)

    range=pow(2,ncontours)
    logrange=log(range)
    normmagz=normalize_array_magnitude(maskzarray, colorpower, logrange, rmax=absmax)
    #zrgb=complex_array_to_rgb(transpose(maskzarray),colorpower,logrange,theme,absmax)
    #zmin,zmax=zrgbrange(xarray,yarray,zrgb,xlo,xhi,ylo,yhi,xmultfactor,ymultfactor)
    
    xmin=real(xarray*xmultfactor).min()
    if(xlo==None):
        xloval=xmin
    else:
        xloval=xlo

    xmax=real(xarray*xmultfactor).max()
    if(xhi==None):
        xhival=xmax
    else:
        xhival=xhi

    ymin=real(yarray*ymultfactor).min()
    if(ylo==None):
        yloval=ymin
    else:
        yloval=ylo

    ymax=real(yarray*ymultfactor).max()
    if(yhi==None):
        yhival=ymax
    else:
        yhival=yhi

    print("xmin,xmax,ymin,ymax\t"+str((xmin,xmax,ymin,ymax)))
    #plt.imshow(zrgb,interpolation=interp,origin='lower',aspect='auto',extent=[xmin,xmax,ymin,ymax],alpha=densityalpha)
    #plt.imshow(real(zarray),vmin=zarray.min(),vmax=zarray.max(),extent=[xarray.min(),xarray.max(),yarray.min(),yarray.max()],origin='lower')
#    if(densityalpha>0):
#        plt.set_cmap('hsv')
#        cbar=plt.colorbar(ticks=[0,.5,1])
#        cbar.ax.set_yticklabels(['0','$\pi$','$2\pi$'],fontsize=inpfontsize)
    dtick=log(10)/log(2)/(ncontours+1)
    nmaxtick=int(floor(log(2)/log(10)*(ncontours+1)))
    ticklist=list(arange(1,0,-dtick))
    print("ticklist\t"+str(ticklist))
    ticklabels=logticklabels(nmaxtick)#list(map(str,-arange(nmaxtick+1)))
    print("ticklabels\t"+str(ticklabels))

    contourlevels=list(arange(1,0,-1/ncontours))[::-1]
    contourcolors=list(map(str,map(contourcmap,contourlevels)))
    if(contourcmap==None):
#        contourplot=plt.contour(transpose(xarray)*xmultfactor,
#                                transpose(yarray)*ymultfactor, transpose(normmagz),
#                                levels=contourlevels, extent=[xmin,xmax,ymin,ymax],
#                                colors=contourcolors, alpha=contouralpha,
#                                interpolation=interp, linewidths=contourlinewidths,
#                                vmin=0,vmax=1)
        contourplot=plt.contour(xarray*xmultfactor,
                                yarray*ymultfactor,normmagz,
                                levels=contourlevels, extent=[xmin,xmax,ymin,ymax],
                                colors=contourcolors, alpha=contouralpha,
                                interpolation=interp, linewidths=contourlinewidths,
                                vmin=0,vmax=1)



    if(contourcmap!=None):
        print("contourcmap!=None")
        print("contourcolors\t"+str(contourcolors))
        CP=plt.contour(transpose(xarray)*xmultfactor,
                    transpose(yarray)*ymultfactor, transpose(normmagz),
                    levels=contourlevels,
                    extent=[xmin,xmax,ymin,ymax], cmap=contourcmap,
                    alpha=contouralpha,
                    interpolation=interp,linewidths=contourlinewidths,vmin=0,vmax=1)
        cbar=plt.colorbar(CP,orientation='vertical')
        cbar.set_clim(0,1)
        if(colorpower=='log'):
            cbar.set_ticks(ticklist)
            cbar.set_ticklabels(ticklabels)

    plt.title(legend)
    ax.set_xlabel(xlabel,fontsize=inpfontsize)
    ax.set_ylabel(ylabel,fontsize=inpfontsize)
    ax.set_xlim(xloval,xhival)
    ax.set_ylim(yloval,yhival)

    return fig

def logticklabels(nmaxtick):
    retlist=["1"]
    for i in range(1,nmaxtick+1):
        #retlist.append("$10^{-"+str(i)+"}$")
        retlist.append(logticklabel(i))
    return retlist

def logticklabel(i):
    return "$10^{-"+str(abs(i))+"}$"

def mapmagnitude(X, colorpower, logrange=6, absmax=None):
    if(absmax!=None):
        maxval=absmax
    else:
        maxval=abs(X).max()
    lognorm=real(log(X))-log(maxval)
    angles=angle(X)
    if(colorpower=='log'):
        lognorm=clip(lognorm,-logrange,0)/logrange+1
        retarray=lognorm*exp(1j*angles)
    else:
        lognorm*=colorpower
        retarray=exp(lognorm+1j*angles)
    return retarray

def imshowplot(xarray, yarray, zarray, colorpower=1., logrange=6, legend="",
               xlabel="", ylabel="",  cmap=None, xmultfactor=1.,
               ymultfactor=1.,  xlo=None, xhi=None, ylo=None, yhi=None,
               inpfontsize=15, densityalpha=1., absmax=None,
               symmetricrange=True, gamma=3, maxzval=None, minzval=None,
               xticks=None, yticks=None, topxticks=None, topxticklabels=[""]):

    if(cmap==None):
        #tmpcmap=divergingcubehelix(gamma=gammaval, rnge=.9, starthue1=60, endhue1=180,
        #                           starthue2=60, endhue2=-60)
        tmpcmap=diverginghsv()
        #tmpcmap=divergingcubehelix(gamma=gamma,rnge=.8)
    else:
        tmpcmap=cmap

    #print("showcontours?\t"+str(showcontours))
    interp="bilinear"#"none"#"bilinear"#'none'#"gaussian"#"bilinear"#"none"#"bilinear"
    print("xmultfactor, ymultfactor\t"+str(xmultfactor)+"\t"+str(ymultfactor))
    #[pltx,plty,pltz]=pltarrays(xarray,yarray,zarray)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.axes[0].set_axis_bgcolor("black")


    maskzarray=maskarray(xarray, yarray, zarray, xlo, xhi, ylo, yhi,
                         xmultfactor, ymultfactor)
    #maskzarray=real(normalize_colorpower(maskzarray,colorpower=colorpower,logrange=logrange))
    if(colorpower=='log'):
        maskzarray=real(mapmagnitude(maskzarray,logrange=logrange,
                                     colorpower=colorpower, absmax=absmax))
    #print("maskzarray\t"+str(maskzarray))
    
    xmin=real(xarray*xmultfactor).min()
    if(xlo==None):
        xloval=xmin
    else:
        xloval=xlo

    xmax=real(xarray*xmultfactor).max()
    if(xhi==None):
        xhival=xmax
    else:
        xhival=xhi

    ymin=real(yarray*ymultfactor).min()
    if(ylo==None):
        yloval=ymin
    else:
        yloval=ylo

    ymax=real(yarray*ymultfactor).max()
    if(yhi==None):
        yhival=ymax
    else:
        yhival=yhi

    print("xmin,xmax,ymin,ymax\t"+str((xmin,xmax,ymin,ymax)))
#unnecessary given norm function already applied
#    normval=abs(maskzarray).max()
#    if(absmax!=None):
#        normval=absmax
#    maskzarray/=normval
    if(maxzval==None and minzval==None):
        if(symmetricrange):
            maxzval=abs(maskzarray).max()
            #maxzval=round(maxzval,2)#round_sig(maxzval,sig=2)
            minzval=-maxzval
        else:
            minzval=maskzarray.min()
            print("minzval\t"+str(minzval))
            #minzval=round(minzval,2)#round_sig(minzval,sig=2)
            maxzval=maskzarray.max()
            #maxzval=round(maxzval,2)#round_sig(maxzval,sig=2)

    print("maxzval used in imshow\t"+str(maxzval)+"\t"+str(minzval))    
    
    #plt.imshow(zrgb,interpolation=interp,origin='lower',aspect='auto',extent=[xmin,xmax,ymin,ymax],alpha=densityalpha)
    plt.imshow(transpose(maskzarray), interpolation=interp, origin='lower',
               aspect='auto', extent=[xmin,xmax,ymin,ymax],
               alpha=densityalpha, cmap=tmpcmap, vmax=maxzval, vmin=minzval)
    if(xticks!=None):
        plt.xticks(xticks, fontsize=inpfontsize)
    else:
        plt.xticks(fontsize=inpfontsize)
    if(yticks!=None):
        plt.yticks(yticks, fontsize=inpfontsize)
    else:
        plt.yticks(fontsize=inpfontsize)
    if(colorpower=='log'):
#        tenpower=int(floor(logrange/log(10)))
#        ticklist=list(range(-tenpower,0))+list(range(tenpower+1))
#        ticklist=array(ticklist)
#        tickpts=array(ticklist)/logrange*log(10)
#        ticklabels=list(map(logticklable,ticklist))
#        cbar=plt.colorbar(ticks=tickpts)
#        inpfontsize=10
#        cbar.ax.set_yticklabels(ticklabels,fontsize=inpfontsize)
        tenpower=int(floor(logrange/log(10)))
        tickarray=1-array(range(tenpower+1))/logrange*log(10)
        indxlist=list(range(tenpower+1))
        ticklist=list(tickarray)+list(-tickarray)
        tickindxlist=indxlist+list(-array(indxlist))
        ticklabellist=list(map(logticklabel,tickindxlist))
        cbar=plt.colorbar(ticks=ticklist)
        cbar.ax.set_yticklabels(ticklabellist,fontsize=inpfontsize)

    else:
        ticks=array([-1,-.5,0,.5,1])*maxzval
        roundedticks=around(ticks,3)
        ticklabels=list(map(str_scientific, ticks))
        print("ticklabels\t"+str(ticklabels))
        #ticks=list(around(array([-1,-.5,0,.5,1])*maxzval,3))
        cbar=plt.colorbar(ticks=ticks)
        #cbar.ax.set_yticklabels(labels=list(map(str,ticks)), fontsize=inpfontsize)
        cbar.ax.set_yticklabels(labels=ticklabels, fontsize=inpfontsize)
    plt.title(legend)
    ax.set_xlabel(xlabel,fontsize=inpfontsize)
    ax.set_ylabel(ylabel,fontsize=inpfontsize)
    ax.set_xlim(xloval,xhival)
    ax.set_ylim(yloval,yhival)
    if(topxticks!=None):
        ax2=ax.twiny()
        ax2.set_xticks(topxticks)
        ax2.set_xticklabels(topxticklabels)

    #plt.imshow(real(zarray),vmin=zarray.min(),vmax=zarray.max(),extent=[xarray.min(),xarray.max(),yarray.min(),yarray.max()],origin='lower')
    return fig

################################################################################
def imshowplot_hsv(xarray,yarray,zarray,colorpower=1.,
                   logrange=6,legend="",xlabel="",ylabel="",
                   theme='hsv',xmultfactor=1.,ymultfactor=1.,
                   xlo=None,xhi=None,ylo=None,yhi=None,
                   inpfontsize=15,absplot=False,absmax=None,showcontours=False,
                   ncontours=5, contouralpha=1.,densityalpha=1.,
                   contourcolors='w', contourcmap=None, contourlinewidths=2):
    #print("showcontours?\t"+str(showcontours))
    interp="bilinear"#"none"#"bilinear"#'none'#"gaussian"#"bilinear"#"none"#"bilinear"
    print("xmultfactor, ymultfactor\t"+str(xmultfactor)+"\t"+str(ymultfactor))
    #[pltx,plty,pltz]=pltarrays(xarray,yarray,zarray)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.axes[0].set_axis_bgcolor("black")
    maskzarray=maskarray(xarray,yarray,zarray,xlo,xhi,ylo,yhi,
                         xmultfactor,ymultfactor)
    zrgb=complex_array_to_rgb(transpose(maskzarray),colorpower,logrange,theme,absmax)
    #zmin,zmax=zrgbrange(xarray,yarray,zrgb,xlo,xhi,ylo,yhi,xmultfactor,ymultfactor)
    
    xmin=real(xarray*xmultfactor).min()
    if(xlo==None):
        xloval=xmin
    else:
        xloval=xlo

    xmax=real(xarray*xmultfactor).max()
    if(xhi==None):
        xhival=xmax
    else:
        xhival=xhi

    ymin=real(yarray*ymultfactor).min()
    if(ylo==None):
        yloval=ymin
    else:
        yloval=ylo

    ymax=real(yarray*ymultfactor).max()
    if(yhi==None):
        yhival=ymax
    else:
        yhival=yhi

    print("xmin,xmax,ymin,ymax\t"+str((xmin,xmax,ymin,ymax)))
    plt.imshow(zrgb,interpolation=interp,origin='lower',aspect='auto',extent=[xmin,xmax,ymin,ymax],alpha=densityalpha)
    #plt.imshow(real(zarray),vmin=zarray.min(),vmax=zarray.max(),extent=[xarray.min(),xarray.max(),yarray.min(),yarray.max()],origin='lower')
    if(densityalpha>0):
        plt.set_cmap('hsv')
        cbar=plt.colorbar(ticks=[0,.5,1])
        cbar.ax.set_yticklabels(['0','$\pi$','$2\pi$'],fontsize=inpfontsize)
    plt.title(legend)
    ax.set_xlabel(xlabel,fontsize=inpfontsize)
    ax.set_ylabel(ylabel,fontsize=inpfontsize)
    ax.set_xlim(xloval,xhival)
    ax.set_ylim(yloval,yhival)

    contourlevels=list(arange(1,0,-1/ncontours))
    if(contourcmap!=None):
        contourcolors=list(map(str,map(contourcmap,contourlevels)))
        print("contourcolors\t"+str(contourcolors))
    if(showcontours):
        print("showcontours\t"+str(showcontours))
        normmagz=normalize_array_magnitude(zarray, colorpower, logrange)
        contourplot=plt.contour(transpose(xarray)*xmultfactor,
                    transpose(yarray)*ymultfactor, transpose(normmagz),
                    levels=contourlevels,# colors=contourcolors,
                    extent=[xmin,xmax,ymin,ymax], cmap=contourcmap,
                    alpha=contouralpha,
                    interpolation=interp,linewidths=contourlinewidths,vmin=0,vmax=1)
#    if(contourcmap!=None):
#        plt.set_cmap(contourcmap)
#        dtick=log(10)/log(2)/ncontours
#        nmaxtick=int(floor(log(2)/log(10)*ncontours))
#        ticklist=list(arange(1,0,-dtick))
#        print("ticklist\t"+str(ticklist))
#        ticklabels=list(map(str,-arange(nmaxtick+1)))
#        print("ticklabels\t"+str(ticklabels))
#        cbar=plt.colorbar(contourplot,orientation='vertical')
#        plt.clim(0,1)
##        cbar.ax.set_yticklabels(ticklabels)
#        cbar.update_ticks()
    return fig

def imshowplot_fun(xarray, yarray, zarray, colorpower=1.,  logrange=6,
                   legend="", xlabel="", ylabel="", theme='hsv',
                   xmultfactor=1., ymultfactor=1., xlo=None, xhi=None,
                   ylo=None,  yhi=None, inpfontsize=15, arrayfunction='logz',
                   absmax=None, showcontours=False, ncontours=5,
                   contouralpha=1., densityalpha=1., contourcolors='w'):
    interp="bilinear"#"gaussian"#"bilinear"#"none"#"bilinear"
    print("xmultfactor, ymultfactor\t"+str(xmultfactor)+"\t"+str(ymultfactor))
    #[pltx,plty,pltz]=pltarrays(xarray,yarray,zarray)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.axes[0].set_axis_bgcolor("black")
    #maskzarray=maskarray(xarray,yarray,zarray,xlo,xhi,ylo,yhi,xmultfactor,ymultfactor)
    #print("zarray\t"+str(zarray))
    if(arrayfunction==logz):
        zplot=arrayfunction(zarray,logrange=logrange,rmax=absmax)#arrayfunction(zarray)
    else:
        zplot=arrayfunction(zarray)
    #print("zplot\t"+str(shape(zplot)))
    #zrgb=complex_array_to_rgb(transpose(maskzarray),colorpower,logrange,theme)
    #zmin,zmax=zrgbrange(xarray,yarray,zrgb,xlo,xhi,ylo,yhi,xmultfactor,ymultfactor)
    
    xmin=real(xarray*xmultfactor).min()
    if(xlo==None):
        xloval=xmin
    else:
        xloval=xlo

    xmax=real(xarray*xmultfactor).max()
    if(xhi==None):
        xhival=xmax
    else:
        xhival=xhi

    ymin=real(yarray*ymultfactor).min()
    if(ylo==None):
        yloval=ymin
    else:
        yloval=ylo

    ymax=real(yarray*ymultfactor).max()
    if(yhi==None):
        yhival=ymax
    else:
        yhival=yhi

    print("xmin,xmax,ymin,ymax\t"+str((xmin,xmax,ymin,ymax)))
    plt.set_cmap('hot')
    plt.imshow(transpose(zplot),interpolation=interp,origin='lower',aspect='auto',extent=[xmin,xmax,ymin,ymax], alpha=densityalpha)
    #plt.imshow(real(zarray),vmin=zarray.min(),vmax=zarray.max(),extent=[xarray.min(),xarray.max(),yarray.min(),yarray.max()],origin='lower')
    if(densityalpha>0):
        cbar=plt.colorbar()
    #cbar.ax.set_yticklabels(['0','$\pi$','$2\pi$'],fontsize=inpfontsize)
    plt.title(legend)
    ax.set_xlabel(xlabel,fontsize=inpfontsize)
    ax.set_ylabel(ylabel,fontsize=inpfontsize)
    ax.set_xlim(xloval,xhival)
    ax.set_ylim(yloval,yhival)

    if(showcontours):
        normmagz=normalize_array_magnitude(zarray, colorpower, logrange)
        plt.contour(transpose(xarray)*xmultfactor,
                    transpose(yarray)*ymultfactor,transpose(normmagz),
                    levels=arange(0,1,1/ncontours)[1:],
                    extent=[xmin,xmax,ymin,ymax], alpha=contouralpha,
                    colors=contourcolors,interpolation=interp)

    return fig


def imshowplotfile(filename,colorpower=1.,logrange=6,legend="",\
                   xlabel="",ylabel="",theme='hsv',\
                   xmultfactor=1.,ymultfactor=1.,xlo=None,xhi=None,\
                   ylo=None,yhi=None,inpfontsize=15,\
                   arrayfunction=None,absplot=False, showcontours=False,
                   ncontours=5, contouralpha=1., densityalpha=1.,
                   contourcolors='w', contourcmap=None, contourlinewidths=2):
    #print("plotfile showcontours?\t"+str(showcontours))
    [xdat,ydat,zdat]=datfromfile(filename)
    xarray,yarray,zarray=dattoarrays(xdat,ydat,zdat)
    if(arrayfunction==None):
        fig=imshowplot_hsv(xarray, yarray, zarray,
                           colorpower=colorpower,logrange=logrange,
                           legend=legend, xlabel=xlabel, ylabel=ylabel,
                           theme=theme, xmultfactor=xmultfactor,
                           ymultfactor=ymultfactor, xlo=xlo, xhi=xhi, ylo=ylo,
                           yhi=yhi, inpfontsize=inpfontsize, absplot=absplot,
                           showcontours=showcontours, ncontours=ncontours,
                           contourcolors=contourcolors,
                           contouralpha=contouralpha,
                           densityalpha=densityalpha, contourcmap=contourcmap,
                           contourlinewidths=contourlinewidths)
    else:
        fig=imshowplot_fun(xarray, yarray, zarray, colorpower=colorpower,
                           logrange=logrange, legend=legend, xlabel=xlabel,
                           ylabel=ylabel, theme=theme, xmultfactor=xmultfactor,
                           ymultfactor=ymultfactor, xlo=xlo, xhi=xhi, ylo=ylo,
                           yhi=yhi, inpfontsize=inpfontsize,
                           arrayfunction=arrayfunction,
                           showcontours=showcontours, ncontours=ncontours,
                           contourcolors=contourcolors,
                           contouralpha=contouralpha,
                           densityalpha=densityalpha)
    return fig



#def contourplotfile(filename,colorpower=1.,logrange=6,legend="",
#                    xlabel="",ylabel="",theme='hsv',
#                    xmultfactor=1.,ymultfactor=1.,xlo=None,xhi=None,
#                    ylo=None,yhi=None,inpfontsize=15, absplot=False,
#                    ncontours=5, contouralpha=1.,
#                    densityalpha=1., contourcolors='w', contourcmap=None,
#                    contourlinewidths=1):
def contourplotfile(filename, **kwargs):
    #print("plotfile showcontours?\t"+str(showcontours))
    #[xdat,ydat,zdat]=datfromfile(filename)
    #xarray,yarray,zarray=dattoarrays(xdat,ydat,zdat)
    xarray,yarray,zarray=arraysfromfile(filename)
    fig=contourplot(xarray,yarray,zarray, **kwargs)
    return fig


#########################################
#Other plotting methods (haven't used in a while; these may not work)
#########################################
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


###################################
#functions to make data arrays more suitable for plotting
def logz(zarray,logrange=6,rmax=None):
    if(rmax==None):
        absmax=abs(zarray).max()
    else:
        absmax=rmax

    retz=array(list(map(lambda x:real(clip(log(x)-log(absmax),-logrange,0)),zarray)))/log(10)
    return retz

def normalize_array_magnitude(X, colorpower=1, logrange=6, rmax=None):
    #maps array magnitudes to range from 0 to 1 in the same way as
    #complex_array_to_rgb
    maxrange=1.
    if(rmax==None):
        absmax=abs(X).max()
    else:
        absmax=rmax
    retarray=[]
    if(colorpower=='log'):
        retarray=pow(clip(log(abs(X))-log(absmax),-logrange,0)/logrange+1,1)
    else:
        retarray= pow(clip(abs(X) / absmax, 0, 1),colorpower)
    return retarray
 
def normalize_colorpower(X,colorpower=1,logrange=6, absmax=None):
    if(absmax==None):
        maxval=abs(X).max()
    else:
        maxval=absmax
    if(colorpower=='log'):
        retarray=clip(log(X)-log(maxval),-logrange,0)
        (n0,n1)=shape(retarray)
        for i,j in product(range(n0), range(n1)):
            if(isinf(retarray[i,j])):
                retarray[i,j]=-logrange
    else:
        retarray = pow(clip(abs(X) / maxval, 0, 1),colorpower)
    return retarray
         

def complex_array_to_rgb(X,colorpower=1,logrange=6, theme='hsv', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    maxrange=1.
    if(not theme in ['light','dark','hls','lab','hsv']):
        print("theme not recognized in complex_array_to_rgb:\t"+theme)
    if(rmax==None):
        absmax=abs(X).max()
    else:
        absmax=rmax
    #absmax = rmax or abs(X).max()
    Y = zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = (angle(X)/ ( 2*pi) % 1)
    if(theme == 'light') or (theme =='hls'):
        if(colorpower=='log'):
            #Y[...,1]=clip(log(X)-log(absmax),-logrange,0)/logrange+1
            Y[...,1]=pow(clip(log(X)-log(absmax),-logrange,0)/logrange+1,1)
            Y[...,2]= Y[...,1]
        else:
            Y[..., 1] = pow(clip(abs(X) / absmax, 0, 1),colorpower)
            Y[..., 2] = Y[...,1]
        #Y[...,1]*=.9
        #Y[..., 2] = 1.#Y[...,1]#1.
        Y[...,1]*=maxrange
        Y[...,2]*=maxrange
    elif(theme == 'dark' or theme =='hsv'):
        Y[..., 1] = 1.
        if(colorpower=='log'):
            Y[...,2]=pow(clip(log(X)-log(absmax),-logrange,0)/logrange+1,1)
            Y[...,1]=Y[...,2]
        else:
            Y[..., 2] = pow(clip(abs(X) / absmax, 0, 1),colorpower)
            Y[..., 1] ==Y[...,2] 
        Y[...,1]*=maxrange
        Y[...,2]*=maxrange
        #Y[...,2]*=.9
        #Y[..., 2] = pow(clip(abs(X) / absmax, 0, 1),colorpower)
        #Y[..., 1]=Y[...,2]
    if(theme=='hls'):
        #use hls instead of hsv
        Y = hls_to_rgb_array(Y)#hsv_to_rgb(Y)
    elif(theme=='hsv'):
        Y=hsv_to_rgb_arry(Y)

    if(theme=='lab'):
        print("theme == lab")
        Y = zeros(X.shape + (3,), dtype='float')
        if(colorpower=='log'):
            Y[...,0]=(clip(log(X)-log(absmax),-logrange,0)/logrange+1)*100
        else:
            Y[..., 0] = pow(clip(abs(X) / absmax, 0, 1),colorpower)*100
        Y[...,1]=(cos(angle(X)))*100
        Y[...,2]=(sin(angle(X)))*100
        Y=luv2rgb(Y)#lab2rgb(Y)
    return Y

def hls_to_rgb_array(Y):
    (nx,ny,nz)=shape(Y)
    for i in range(nx):
        for j in range(ny):
            [h,l,s]=Y[i,j,:]
            #print("h,l,s\t"+str([h,l,s]))
            (r,g,b)=hls_to_rgb(h,l,s)
            #print("r,g,b"+str((r,g,b)))
            Y[i,j,:]=[r,g,b]
    return Y
    

def hsv_to_rgb_arry(Y):
    (nx,ny,nz)=shape(Y)
    for i in range(nx):
        for j in range(ny):
            [h,s,v]=Y[i,j,:]
            #print("h,l,s\t"+str([h,l,s]))
            (r,g,b)=colorsys.hsv_to_rgb(h,s,v)
            #print("r,g,b"+str((r,g,b)))
            Y[i,j,:]=[r,g,b]
    return Y
#################################
#functions to set up a nonlinear color bar, so that smaller values can be
#seen more easily
def lohiindx(x,lst):
    #takes a sorted list as input, returns indices i,j st
    #lst[i]<=x<=lst[j] and j=i+1
    loindx=0
    hiindx=len(lst)-1
    if(x<lst[0]):
        loindx=0
        hiindx=0
    if(x>lst[-1]):
        loindx=hiindx
    while((hiindx-loindx)>1):
        midindx=int(floor((loindx+hiindx)/2))
        if(x>=lst[midindx]):
            loindx=midindx
        else:
            hiindx=midindx
    return loindx,hiindx

def interpcolorarray(x,colorarray):
    #print("interpcolorarray\t"+str(x)+"\n"+str(colorarray))
    loindx,hiindx=lohiindx(x,colorarray[:,0])
    lox=colorarray[loindx,0]
    loxval=colorarray[loindx,1]
    #print("lox,loxval\t"+str(lox)+"\t"+str(loxval))
    hix=colorarray[hiindx,0]
    hixval=colorarray[hiindx,2]
    #print("hix,hixval\t"+str(hix)+"\t"+str(hixval))
    retval=(x-lox)/(hix-lox)*(hixval-loxval)+loxval
    #print("retval\t"+str(x)+"\t"+str(retval))
    return retval

def evalcmap(x,inpcmap):
    inpdict=inpcmap._segmentdata
    #print("inpdict\t"+str(inpdict))
    bluearray=array(inpdict['blue'])
    #print("bluearray\t"+str(bluearray))
    greenarray=array(inpdict['green'])
    redarray=array(inpdict['red'])
    
    blueval=interpcolorarray(x,bluearray)
    redval=interpcolorarray(x,redarray)
    greenval=interpcolorarray(x,greenarray)

    return redval,greenval,blueval


def powerxmap(x,power):
    xtmp=2.*x-1.
    invpower=1./power#use inverse power so that imshow & interp plots
                     #have same color power
    retval= (1+(sign(xtmp)*pow(abs(xtmp),invpower)))/2.
    return retval

def logxmap(x):
    xtmp=2*x-1
    retval=-log(xtmp)*sign(xtmp)

def abspowerxmap(x,power):
    invpower=1./power#use inverse power so that imshow & interp plots
                     #have same color power
    return pow(abs(x),invpower)

def logpowercmap(inpcmapname='coolwarm',logrange=6,nsegments=100):
    bluelist=[]
    redlist=[]
    greenlist=[]
    inpcmap=cm.get_cmap(inpcmapname)
    scalarmap=cm.ScalarMappable(cmap=inpcmap)
    scalarmap.set_clim(0.,1.)
    for i in range(nsegments):
        xi=i*1./(nsegments-1)
        yi=exp((-xi)*logrange)
        (redval,greenval,blueval,alphaval)=scalarmap.to_rgba(xi)
        redlist.append((yi,redval,redval))
        greenlist.append((yi,greenval,greenval))
        bluelist.append((yi,blueval,blueval))
    cdict={'red':redlist,'green':greenlist,'blue':bluelist}
    return cdict

#'coolwarm','rainbow','cubehelix','coolwarm'
def powercmap(power,inpcmapname='hls',nsegments=100,abspowercmapflag=False):
    bluelist=[]
    redlist=[]
    greenlist=[]
    if(inpcmapname=="hls"):
        for i in range(nsegments):
            xi=i*1./(nsegments-1)
            yi=0.
            if(abspowercmapflag):
                yi=abspowerxmap(xi,power)
            else:
                yi=powerxmap(xi,power)
            redval,greenval,blueval=redbluehls(xi)
            redlist.append((yi,redval,redval))
            greenlist.append((yi,greenval,greenval))
            bluelist.append((yi,blueval,blueval))
    else:
        inpcmap=cm.get_cmap(inpcmapname)
        scalarmap=cm.ScalarMappable(cmap=inpcmap)
        scalarmap.set_clim(0.,1.)
        for i in range(nsegments):
            xi=i*1./(nsegments-1)
            #redval,greenval,blueval=redbluecubehelix(xi,0,1,1,1)
            (redval,greenval,blueval,alphaval)=scalarmap.to_rgba(xi)
            yi=0.
            if(abspowercmapflag):
                yi=abspowerxmap(xi,power)
            else:
                yi=powerxmap(xi,power)
            redlist.append((yi,redval,redval))
            greenlist.append((yi,greenval,greenval))
            bluelist.append((yi,blueval,blueval))
    cdict={'red':redlist,'green':greenlist,'blue':bluelist}
    return cdict
#
##this seems to duplicate the "coolwarm" color map
#def redbluecubehelix(x,startdir=0,rotations=1,hue=4.,gamma=1.):
#    if(x>.5):
#        return poscubehelix(2*(x-.5),startdir,rotations,hue,gamma)
#    else:
#        return negcubehelix(2*(.5-x),startdir,rotations,hue,gamma)
#    #return cubehelix(x,startdir,rotations,hue,gamma)
#
#def redbluehls(x):
#    tmpx=(x-.5)
#    h=angle(tmpx)/(2*pi) % 1
#    l=abs(tmpx)
#    s=1.
#    (r,g,b)=hls_to_rgb(h,l,s)
#    return r,g,b
#
#def negcubehelix(x,startdir=0,rotations=1,hue=1,gamma=1):
#    revx=1-x
#    return (1-revx)*array([.891,-.453,0])/sqrt(2.)+revx*cubehelix(revx,startdir+3/4.,rotations,hue,gamma)
#
#def poscubehelix(x,startdir=0,rotations=1,hue=1,gamma=1):
#    revx=1-x
#    return (1-revx)*array([-0.074,-.146,.986])/sqrt(2.)+revx*cubehelix(revx,startdir,rotations,hue,gamma)/sqrt(2.)
#
#def cubehelix(x,startdir,rotations,hue,gamma):
#    #taken from paper defining cubehelix color scheme
#    #x ranges from 0 to 1 (lambda in paper)
#    rgbvec=ones(3)*pow(x,gamma)
#    phival=2*pi*(startdir/3+rotations*x)
#    a=hue*pow(x,gamma)*(1-pow(x,gamma))/2.
#    colormat=transpose(array([[-.14861,-.29227,1.97294],[1.78277,-.90649,0]]))
#    rotvec=array([cos(phival),sin(phival)])
#
#    retrgbvec=rgbvec+a*dot(colormat,rotvec)
#
#    return retrgbvec
#

def transmag(x,gamma):
    y=pow(x,gamma)
    retval=0.
    if(y<.5):
        retval=y
    else:
        retval=1-y
    return retval

def hsvhelixcmap(thetastart=0, nturns=1, reverse=False, bgblack=False):
    sat=1
    if(bgblack):
        retcdict=cfunctocdict(lambda x: hsv2rgb(mod(thetastart/(2*pi)+nturns*x, 1),
                                                sat, x), reverse=reverse)
    else:
        retcdict=cfunctocdict(lambda x: hsv2rgb(mod(thetastart/(2*pi)+nturns*x, 1),
                                                x, sat), reverse=reverse)

    
    retcmap=matplotlib.colors.LinearSegmentedColormap("hsvhelix",retcdict)
    return retcmap

def diverginghsv(thetastart1=pi/3, thetafinish1=2*pi/3,
                        thetastart2=pi/3, thetafinish2=0, gamma=1.,
                        nsegments=101, **kwargs):
    cdict1=cfunctocdict(lambda x: hsvrgb(x, thetastart=thetastart1,
                                         thetafinish=thetafinish1, gamma=1, **kwargs))
    cdict2=cfunctocdict(lambda x: hsvrgb(x, thetastart=thetastart2,
                                         thetafinish=thetafinish2, gamma=1, **kwargs))
    newcdict=combinecdicts(cdict1,cdict2)
      
    tmpcmap=matplotlib.colors.LinearSegmentedColormap("divergingcubehelix",newcdict)
    return tmpcmap

def reverselist(inplist):
    retlist=inplist[::-1]
    for i in range(len(retlist)):
        retlist[i][0]=1-retlist[i][0]
    return retlist

def cfunctocdict(cfunc, nsegments=101, reverse=False):
    dx=1/nsegments
    xarray=arange(0,1+dx,dx)
    redlist=[]
    bluelist=[]
    greenlist=[]
    for i in range(len(xarray)):
        [redval, blueval, greenval]=cfunc(xarray[i]) 
        redval=min(redval,1.)
        redval=max(redval,0.)
        blueval=min(blueval,1.)
        blueval=max(blueval,0.)
        greenval=min(greenval,1.)
        greenval=max(greenval,0.)
        #redlist.append((xarray[i],redval,redval))
        #bluelist.append((xarray[i],blueval,blueval))
        #greenlist.append((xarray[i],greenval,greenval))
        redlist.append([xarray[i],redval,redval])
        bluelist.append([xarray[i],blueval,blueval])
        greenlist.append([xarray[i],greenval,greenval])
#    print("redlist\t"+str(redlist))
    if(reverse):
        redlist=reverselist(redlist)
        bluelist=reverselist(bluelist)
        greenlist=reverselist(greenlist)
    retcdict={'red':redlist, 'green':greenlist, 'blue':bluelist}
    return retcdict

def combinecdicts(cdict1,cdict2):
    red1=cdict1['red']
    blue1=cdict1['blue']
    green1=cdict1['green']
    red2=cdict2['red']
    blue2=cdict2['blue']
    green2=cdict2['green']

    for i in range(len(red1)):
        red1[i][0]=(1-red1[i][0])/2
        blue1[i][0]=(1-blue1[i][0])/2
        green1[i][0]=(1-green1[i][0])/2

    for i in range(len(red2)):
        red2[i][0]=red2[i][0]/2+0.5
        blue2[i][0]=blue2[i][0]/2+0.5
        green2[i][0]=green2[i][0]/2+0.5
    retred=red1[::-1]+red2
    retblue=blue1[::-1]+blue2
    retgreen=green1[::-1]+green2
    retcdict={'red':retred, 'blue':retblue, 'green':retgreen}
    return retcdict


#def z_cubehelix(x, thetastart=0, thetafinish=pi/2, gamma=1,  transfrac=1, reverse=False):
#    if reverse:
#        tmpx=1-x
#    else:
#        tmpx=x
##return [R,G,B]
#    vinitial=array([ 0.33105108, -0.31328768,  0.74494924])
#    v0=array([0.44723310587634052, 0.87801281989769864, 0.17051697013345427])#perceived intensity
#
#    v1=array([-0.        ,  0.19064585, -0.98165888])#first null intensity vector (orthogonal to red)
#    v2=array([0.89441743554573105, -0.43903035074724284, -0.085263134557386169])# second null intensity vector (orthogonal to first two)
#
#    thetaval=thetastart+(thetafinish-thetastart)*tmpx
#    tm=transmag(tmpx,gamma)
#    #return pow(tmpx,gamma)*1.4957*v0#+v1*sin(thetaval)*tm+v2*cos(thetaval)*tm
#    return v1*sin(thetaval)*tm+v2*cos(thetaval)*tm

def hsvrgb(x,thetastart=0,thetafinish=4*pi/3,gamma=1,reverse=False,
           inpdegrees=False, hls=False, bgblack=False, **kwargs):
    tmpx=pow(x,gamma)
    if(reverse):
        tmpx=1-tmpx
    val=tmpx
    hue=(thetastart+(thetafinish-thetastart)*x)
    if(inpdegrees):
        hue=mod(hue,360)/360
    else:
        hue=mod(hue,2*pi)/(2*pi)
    sat=1.
    #[[[r,g,b]]]=hsv_to_rgb(array([[[hue,sat,val]]]))
    if(hls):
        if(bgblack):
            (r,g,b)=hls_to_rgb(hue,sat,val)
        else:
            (r,g,b)=hls_to_rgb(hue,val,sat)
    else:
        if(bgblack):
            (r,g,b)=hsv2rgb(hue,sat,val)
        else:
            (r,g,b)=hsv2rgb(hue,val,sat)
    return [r,g,b]

def combinefuncs(x,f1,f2, rnge=.8):
    if(x>.5):
        return f2((1.-2*(x-.5))*rnge+(1-rnge))
    else:
        return f1(2*x*rnge+(1-rnge))

def cdictfuncstosegments(cdict1,cdict2,nsegments=101, rnge=1.):
    dx=1./nsegments
    xarray=arange(0,1+dx,dx)
    redlist=[]
    bluelist=[]
    greenlist=[]
    red1=cdict1['red']
    red2=cdict2['red']
    blue1=cdict1['blue']
    blue2=cdict2['blue']
    green1=cdict1['green']
    green2=cdict2['green']
    
    for i in range(len(xarray)):
        redval=combinefuncs(xarray[i],red1,red2,rnge=rnge)
        redval=min(redval,1.)
        redval=max(redval,0.)
        blueval=combinefuncs(xarray[i],blue1,blue2,rnge=rnge)
        blueval=min(blueval,1.)
        blueval=max(blueval,0.)
        greenval=combinefuncs(xarray[i],green1,green2,rnge=rnge)
        greenval=min(greenval,1.)
        greenval=max(greenval,0.)
        redlist.append((xarray[i],redval,redval))
        bluelist.append((xarray[i],blueval,blueval))
        greenlist.append((xarray[i],greenval,greenval))
#    print("redlist\t"+str(redlist))
    retcdict={'red':redlist, 'green':greenlist, 'blue':bluelist}
    return retcdict

def cmap(start=0.5, rot=-1.5, gamma=1.0, reverse=False, nlev=256.,
         minSat=1.2, maxSat=1.2, minLight=0., maxLight=1.,
         **kwargs):
    """
    A full implementation of Dave Green's "cubehelix" for Matplotlib.
    Based on the FORTRAN 77 code provided in
    D.A. Green, 2011, BASI, 39, 289.
    http://adsabs.harvard.edu/abs/2011arXiv1108.5083G
    User can adjust all parameters of the cubehelix algorithm.
    This enables much greater flexibility in choosing color maps, while
    always ensuring the color map scales in intensity from black
    to white. A few simple examples:
    Default color map settings produce the standard "cubehelix".
    Create color map in only blues by setting rot=0 and start=0.
    Create reverse (white to black) backwards through the rainbow once
    by setting rot=1 and reverse=True.
    Parameters
    ----------
    start : scalar, optional
        Sets the starting position in the color space. 0=blue, 1=red,
        2=green. Defaults to 0.5.
    rot : scalar, optional
        The number of rotations through the rainbow. Can be positive
        or negative, indicating direction of rainbow. Negative values
        correspond to Blue->Red direction. Defaults to -1.5
    gamma : scalar, optional
        The gamma correction for intensity. Defaults to 1.0
    reverse : boolean, optional
        Set to True to reverse the color map. Will go from black to
        white. Good for density plots where shade~density. Defaults to False
    nlev : scalar, optional
        Defines the number of discrete levels to render colors at.
        Defaults to 256.
    sat : scalar, optional
        The saturation intensity factor. Defaults to 1.2
        NOTE: this was formerly known as "hue" parameter
    minSat : scalar, optional
        Sets the minimum-level saturation. Defaults to 1.2
    maxSat : scalar, optional
        Sets the maximum-level saturation. Defaults to 1.2
    startHue : scalar, optional
        Sets the starting color, ranging from [0, 360], as in
        D3 version by @mbostock
        NOTE: overrides values in start parameter
    endHue : scalar, optional
        Sets the ending color, ranging from [0, 360], as in
        D3 version by @mbostock
        NOTE: overrides values in rot parameter
    minLight : scalar, optional
        Sets the minimum lightness value. Defaults to 0.
    maxLight : scalar, optional
        Sets the maximum lightness value. Defaults to 1.
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap object
    Example
    -------
    >>> import cubehelix
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> y = np.random.randn(1000)
    >>> cx = cubehelix.cmap(start=0., rot=-0.5)
    >>> plt.hexbin(x, y, gridsize=50, cmap=cx)
    Revisions
    ---------
    2014-04 (@jradavenport) Ported from IDL version
    2014-04 (@jradavenport) Added kwargs to enable similar to D3 version,
                            changed name of "hue" parameter to "sat"
    """

# override start and rot if startHue and endHue are set
    if kwargs is not None:
        if 'startHue' in kwargs:
            start = (kwargs.get('startHue') / 360. - 1.) * 3.
        if 'endHue' in kwargs:
            rot = kwargs.get('endHue') / 360. - start / 3. - 1.
        if 'sat' in kwargs:
            minSat = kwargs.get('sat')
            maxSat = kwargs.get('sat')

# set up the parameters
    fract = np.linspace(minLight, maxLight, nlev)
    angle = 2.0 * pi * (start / 3.0 + rot * fract + 1.)
    fract = fract**gamma

    satar = np.linspace(minSat, maxSat, nlev)
    amp = satar * fract * (1. - fract) / 2.

# compute the RGB vectors according to main equations
    red = fract + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
    grn = fract + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
    blu = fract + amp * (1.97294 * np.cos(angle))

# find where RBB are outside the range [0,1], clip
    red[np.where((red > 1.))] = 1.
    grn[np.where((grn > 1.))] = 1.
    blu[np.where((blu > 1.))] = 1.

    red[np.where((red < 0.))] = 0.
    grn[np.where((grn < 0.))] = 0.
    blu[np.where((blu < 0.))] = 0.

# optional color reverse
    if reverse is True:
        red = red[::-1]
        blu = blu[::-1]
        grn = grn[::-1]

# put in to tuple & dictionary structures needed
    rr = []
    bb = []
    gg = []
    for k in range(0, int(nlev)):
        rr.append((float(k) / (nlev - 1.), red[k], red[k]))
        bb.append((float(k) / (nlev - 1.), blu[k], blu[k]))
        gg.append((float(k) / (nlev - 1.), grn[k], grn[k]))

    cdict = {'red': rr, 'blue': bb, 'green': gg}
    return LSC('cubehelix_map', cdict)


#def divergingcubehelix(s1=0.6, r1=-0.5, s2=0, r2=0.5, saturation=4., gamma=1.,nsegments=101,rnge=1., **kwargs):
def divergingcubehelix(starthue1=60, endhue1=180, starthue2=60, endhue2=-60,
                       saturation=4, gamma=1., nsegments=101, rnge=.9,
                       **kwargs):
    #tmpcdict1=cubehelix(gamma=gamma,s=2,r=rot,h=hue)
    #tmpcdict2=cubehelix(gamma=gamma,s=1,r=-rot,h=hue)
    s1 = ((starthue1) / 360. - 1.) * 3.
    r1 = endhue1 / 360. - s1 / 3. - 1.
    s2 = (starthue2 / 360. - 1.) * 3.
    r2 = endhue2 / 360. - s2 / 3. - 1.
 

    tmpcdict1=cubehelix(gamma=gamma, s=s1, r=r1, h=saturation)
    tmpcdict2=cubehelix(gamma=gamma, s=s2, r=r2, h=saturation)
    tmpcdict3list=cdictfuncstosegments(cdict1=tmpcdict1, cdict2=tmpcdict2,
                                       nsegments=nsegments, rnge=rnge)
    

    tmpcmap=matplotlib.colors.LinearSegmentedColormap("divergingcubehelix",tmpcdict3list)
    return tmpcmap


##'coolwarm','seismic'
#def powercmap(power,inpcmapname='rainbow',nsegments=100,abspowercmapflag=False):
#    inpcmap=cm.get_cmap(inpcmapname)
#    bluelist=[]
#    redlist=[]
#    greenlist=[]
#    for i in range(nsegments):
#        xi=i*1./(nsegments-1.)
#        redval,greenval,blueval=evalcmap(xi,inpcmap)
#        yi=0.
#        if(abspowercmapflag):
#            yi=abspowerxmap(xi,power)
#        else:
#            yi=powerxmap(xi,power)
#        #print("xi,yi,redval,greenval,blueval\t"+str(xi)+"\t"+str(yi)+"\t"+str((redval,greenval,blueval)))
#        redlist.append((yi,redval,redval))
#        greenlist.append((yi,greenval,greenval))
#        bluelist.append((yi,blueval,blueval))
#    cdict={'red':redlist,'green':greenlist,'blue':bluelist}
#    return cdict

#'hot','autumn'
def abspowercmap(power,inpcmapname='hot',nsegments=100,abspowercmapflag=True):
    return powercmap(power,inpcmapname,nsegments,abspowercmapflag)
#def abspowercmap(power):
#    #want blue to increase from 0 to 1
#    nsegments=10
#    bluemin=0
#    bluemax=1.
#    bluelist=[(0,0,0)]
#    for i in range(nsegments):
#        xi=bluemin+(i/(nsegments-1))*(bluemax-bluemin)
#        #yi=pow(abs(xi-bluemin),power)
#        yi=pow(abs(i/(nsegments-1)),power)
#        bluelist.append((xi,yi,yi))
#    redlist=[(0,0,0),(1,0,0)]
#    greenlist=[(0,0,0),(1,0,0)]
#    cdict={'red':redlist,'green':greenlist,'blue':bluelist}
#    #print('cdict\t'+str(cdict))
#    return cdict
#
#def powercmap(power):
#    #want red to increase from 0 to 1 over the bottom half,
#    #blue to increase from 0 to 1 over the top half
#    nsegments=10
#    redmin=0.
#    redmax=0.5
#    redlist=[]
#    for i in range(nsegments):
#        xi=redmin+(i/(nsegments-1))*(redmax-redmin)
#        #yi=pow(abs(redmax-xi),power)
#        yi=pow(1-(i/(nsegments-1)),power)
#        redlist.append((xi,yi,yi))
#    redlist.append((1.,0,0))
#
#    bluemin=0.5
#    bluemax=1.
#    bluelist=[(0,0,0)]
#    for i in range(nsegments):
#        xi=bluemin+(i/(nsegments-1))*(bluemax-bluemin)
#        #yi=pow(abs(xi-bluemin),power)
#        yi=pow(abs(i/(nsegments-1)),power)
#        bluelist.append((xi,yi,yi))
#    greenlist=[(0,0,0),(1,0,0)]
#
#    cdict= {'red': redlist,'green':greenlist,'blue':bluelist}
#    #cdict = {'red':   [(0.0,1.0,1.0),(0.5,0.0,0.0),(1.0,0.0,0.0)],
#    #     'green': [(0.0,0.0,0.0),(1.0,0.0,0.0)],
#    #     'blue':  [(0.0,0.0,0.0),(0.5,0.0,0.0),(1.0,1.0,1.0)]}
#    #print("cdict\t"+str(cdict))
#    return cdict
    

##############################
def lognormval(val,maxval,minratio):
    #want to map (maxval*minratio,maxval) to (0,1) logarithmically
    sgn=sign(val)
    logabsratio=log(abs(val/maxval))
    minlogabsratio=log(minratio)
    return sgn*(1-logabsratio/minlogabsratio)
    

def lognorm(zdat,minratio):
    maxval=abs(zdat).max()
    retdat=zdat*1.
    n=len(zdat)
    for i in range(n):
        retdat[i]=lognormval(zdat[i],maxval,minratio)
    return retdat

def pownorm(zdat,power):
    maxval=abs(zdat).max()
    retdat=zdat*1.
    for i in range(len(zdat)):
        retdat[i]=sign(zdat[i])*pow(abs(zdat[i]/maxval),power)
    return retdat


def vecstoarrays(vec1,vec2):
    n=len(vec1)
    m=len(vec2)
    retarray1=outer(vec1,ones(m))
    retarray2=outer(ones(n),vec2)
    return retarray1,retarray2


