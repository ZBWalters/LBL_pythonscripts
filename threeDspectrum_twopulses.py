from numpy import *
from numpy.linalg import matrix_power,solve
import os
import sys
import subprocess
sys.path.append(os.path.expanduser("~/pythonscripts/"))
from twoDplot import *
from itertools import product, chain
from numpy.fft import fftn,ifftn,fftshift
from scipy.signal import deconvolve



#read in arrays
phasevec=load("phasevec.pkl")
print("shape phasevec\t"+str(shape(phasevec)))
deltatvec=load("deltatvec.pkl")
print("shape deltatvec\t"+str(shape(deltatvec)))
tmeasurevec=load("tmeasurevec.pkl")
print("shape tmeasurevec\t"+str(shape(tmeasurevec)))
woutvec=load("woutvec.pkl")
print("shape woutvec\t"+str(shape(woutvec)))
nprvec=load("nprvec.pkl")
print("shape nprvec\t"+str(shape(nprvec)))
nsvec=load("nsvec.pkl")
print("shape nsvec\t"+str(shape(nsvec)))
deltat_npr_ns_tmeasure_array=load("deltat_npr_ns_tmeasure_array.pkl")
print("shape deltat_npr_ns_tmeasure_array\t"+str(shape(deltat_npr_ns_tmeasure_array)))
deltat_npr_ns_wout_array=load("deltat_npr_ns_wout_array.pkl")
print("shape deltat_npr_ns_wout_array\t"+str(shape(deltat_npr_ns_wout_array)))
deltat_npr_phase_tmeasure_array=load("deltat_npr_phase_tmeasure_array.pkl")
print("shape deltat_npr_phase_tmeasure_array\t"+str(shape(deltat_npr_phase_tmeasure_array)))
deltat_npr_phase_wout_array=load("deltat_npr_phase_wout_array.pkl")
print("shape deltat_npr_phase_wout_array\t"+str(shape(deltat_npr_phase_wout_array)))

####################################################################
#array operations

def iftarray(inparray,axes,shift=True):
    if(shift):
        retarray=fftshift(inparray,axes=axes)
    else:
        retarray=copy(inparray)
    retarray=ifftn(retarray,axes=axes)
    return retarray


def ftarray(inparray,axes,shift=True):
    retarray=fftn(inparray,axes=axes)
    if(shift):
        retarray=fftshift(retarray,axes=axes)
    return retarray


################################################################################
#Movie tools

#figure tools
def fign(n,colorpower='log',logrange=6,inpstr="kIR"):
    return imshowplotfile("phasematched_multidimensional/"+str(n)+
                          "/freq_vs_freq_"+inpstr+".dat",
                          arrayfunction=logz,xlo=0,ymultfactor=Hrt,
                          colorpower=colorpower,logrange=logrange,ylo=0.,yhi=25)

#array operations
def abssqarraymean(inparray):
    (n0,n1,n2)=shape(inparray)
    retarray=zeros((n0,n2))
    for i,j in product(range(n0),range(n2)):
        retarray[i,j]=mean(abs(inparray[i,:,j])**2)
    return retarray

def normalizearray(inparray):
    retarray=copy(inparray)
    arraymean=abssqarraymean(inparray)
    (n0,n1,n2)=shape(inparray)
    for i,j in product(range(n0),range(n2)):
        retarray[i,:,j]/=arraymean[i,j]
    return retarray
    
def zerooriginalarrayphase(inparray):
    retarray=copy(inparray)
    (n0,n1,n2)=shape(inparray)
    for i,j in product(range(n0),range(n2)):
        tmpangle=angle(retarray[i,0,j])
        retarray[i,:,j]*=exp(-1j*tmpangle)
    return retarray
        

def subtractarraymeansq(inparray):
    retarray=copy(inparray)
    arraymean=abssqarraymean(inparray)
    (n0,n1,n2)=shape(inparray)
    for i,j in product(range(n0),range(n2)):
        retarray[i,:,j]=(abs(retarray[i,:,j])**2-arraymean[i,j])
    return retarray

def zeroarraycolumns(inparray,colnumlist):
    retarray=copy(inparray)
    for colnum in colnumlist:
        retarray[:,colnum,:]=0.
    return retarray

def zeroarraycolumn(inparray,colnum):
    return zeroarraycolumns(inparray,[colnum])
#    retarray=copy(inparray)
#    retarray[:,colnum,:]=0.
#    return retarray
def zeroftindices(inparray,colnumlist):
    (n0,n1,n2)=shape(inparray)
    indxlist=ftindexlist(n1,colnumlist)
    return zeroarraycolumns(inparray,indxlist)

def ftindexlist(nft,colnumlist):
    nftby2=int(floor(nft/2))
    indxlist=list(range(-nftby2,-nftby2+nft))
    retlist=[]
    for i in range(len(colnumlist)):
        retlist.append(indxlist.index(colnumlist[i]))
    return retlist

def vecstoarrays(vec1,vec2):
    n=len(vec1)
    m=len(vec2)
    retarray1=outer(vec1,ones(m))
    retarray2=outer(ones(n),vec2)
    return retarray1,retarray2
    
def iftindx1(inparray):
    iftarray=ifftn(inparray,axes=[1])
    iftarray=fftshift(iftarray,axes=[1])
    return iftarray

#def arraymovie(inparray, movietitle="movie.mp4", colorpower='log', logrange=6,
#               arrayfun=None, xvec=winpvec, yvec=woutvec,
#               removefigs=True, theme='hls', ylo=None, yhi=None):
#    (n0,n1,n2)=shape(wkIRwarray)
#    print("shape wkIRwarray\t"+str(shape(wkIRwarray)))
#    xarray,yarray=vecstoarrays(xvec,yvec)
#
#    zmax=(abs(inparray)).max()
#    fignamelist=[]
#    for n in range(shape(inparray)[1]):
#        if(arrayfun==None):
#            fign=imshowplot_hsv(xarray, yarray, inparray[:, n, :],
#                                colorpower=colorpower, logrange=logrange,
#                                legend="$\phi=$"+"{0:.2f}".format(phasevec[n])+"$\pi$",
#                                ymultfactor=Hrt, xmultfactor=Hrt,
#                                absmax=zmax,  xlabel="$\omega_{in}$ (eV)",
#                                ylabel="$\omega_{out}$ (eV)",  theme=theme,
#                                ylo=ylo,  yhi=yhi)
#        else:
#            fign=imshowplot_fun(xarray, yarray, inparray[:, n, :],
#                                colorpower=colorpower, logrange=logrange,
#                                legend="$\phi=$"+"{0:.2f}".format(phasevec[n])+"$\pi$",
#                                ymultfactor=Hrt, xmultfactor=Hrt,
#                                arrayfunction=arrayfun, absmax=zmax,
#                                xlabel="$\omega_{in}$ (eV)",
#                                ylabel="$\omega_{out}$ (eV)", theme=theme,
#                                ylo=ylo,  yhi=yhi)
#        figname="fig"+"%03d" % n+".png"
#        fign.savefig(figname)
#        plt.close(fign)
#        fignamelist.append(figname)
##        fign=interpplot(winparray.flatten(),woutarray.flatten(),
##                   real(iftnoconstarray[:,n,:].flatten()),
##                        legend="$\phi=$"+"{0:.2f}".format(phasevec[n])+"$\pi$",
##                        colorpower=1,
##                   inpcmapname='RdBu',ylo=0,ymultfactor=Hrt,
##                   xmultfactor=Hrt,zmax=zmax)
#    print(" ".join(fignamelist))
#    subprocess.call(["ffmpeg", "-r", "3",
#                     "-i","fig%03d.png","-pix_fmt", "yuv420p",movietitle])
#    print("created "+movietitle)
#    if(removefigs):
#        figfiles=glob.glob("fig*.png")
#        for figfile in figfiles:
#            os.remove(figfile)
#




####################################################################
#Montage tools
def plotarrayslice(xvec, yvec, arrayslice, plotstyle='density', **kwargs):
    #**kwargs is passed to imshowplot_hsv, so look to that definition for the list
    #of adjustable options
    xarray,yarray=vecstoarrays(xvec,yvec)
    retfig=None
    print("plotstyle\t"+plotstyle)
    if(plotstyle=='density'):
        retfig=imshowplot_hsv(xarray,yarray,arrayslice, **kwargs)
    if(plotstyle=='contour'):
        print("contour plot")
        contourcmapname='spectral'
        retfig=contourplot(xarray, yarray, arrayslice, contourcmapname=contourcmapname,
                           **kwargs)
    return retfig
    
def arraymontage(xvec, yvec, tvec, inparray, indxorder=[0,1,2,3],
                 lastindxval=0, colorpower='log', logrange=6,
                 nrows=None, montagetitle=None, arraynormalize=True,
                 removefigs=True, plotstyle='contour', **kwargs):
    #This routine reshapes inparray so that the indices appear in the order
    #specified by indxorder.  It then makes an array montage of x vs y plots by
    #looping through t, where the (rearranged) zeroth index is the x axis, first
    #index is the y axis, second is the t axis.  The last index value is
    #specified by lastindxval.
    print("arraymontage plotstyle\t"+plotstyle)
    tmparray=inparray.transpose(indxorder)
    if(arraynormalize):
        absmax=abs(tmparray[:,:,:,lastindxval]).max()
    else:
        absmax=None
    print("shape inparray\t"+str(shape(inparray)))
    print("shape tmparray\t"+str(shape(tmparray)))
    fignamelist=[]
    for i in range(len(tvec)):
        tval=tvec[i]
        fign=plotarrayslice(xvec,yvec,tmparray[:,:,i,lastindxval],
                            absmax=absmax, colorpower=colorpower,
                            logrange=logrange, plotstyle=plotstyle, **kwargs)
        figname="fig"+"%03d" % i+".png"
        fign.savefig(figname)
        plt.close(fign)
        fignamelist.append(figname)
    makemontage(fignamelist,nrows,montagetitle)
    if(removefigs):
        deletefiglist(fignamelist)

#def arraymontage(inparray, indxlist=kIRvec, montagetitle="montage.png",
#                 colorpower='log', logrange=6, arrayfun=None,
#                 xvec=winpvec, yvec=woutvec, removefigs=True,
#                 nrows=None, nmin=None, nmax=None, nstride=1,  ylo=None, yhi=None,
#                 xlabel="$\omega_{in}$ (eV)", ylabel="$\omega_{out}$ (eV)",
#                 arraynormalize=True):
#    (n0,n1,n2)=shape(inparray)
#    if(arraynormalize):
#        zmax=abs(inparray).max()
#    else:
#        zmax=None
#    fignamelist=[]
#
#    xarray,yarray=vecstoarrays(xvec,yvec)
#
#    loopnmin,loopnmax=indxrange(indxlist,nmin,nmax)
#    for n in range(loopnmin,loopnmax,nstride):
#        if(arrayfun==None):
#            fign=imshowplot_hsv(xarray,yarray,inparray[:,n,:],
#                                colorpower=colorpower,logrange=logrange,
#                                legend="$n_{streak}=$"+indxstr(n,indxlist),
#                                ymultfactor=Hrt, xmultfactor=Hrt, absmax=zmax,
#                                xlabel=xlabel, ylabel=ylabel,ylo=ylo,yhi=yhi)
#        else:
#            fign=imshowplot_fun(xarray,yarray,inparray[:,n,:],
#                                colorpower=colorpower,logrange=logrange,
#                                legend="$n_{streak}=$"+indxstr(n,indxlist),
#                                ymultfactor=Hrt, xmultfactor=Hrt,
#                                arrayfunction=arrayfun, absmax=zmax,
#                                xlabel=xlabel, ylabel=ylabel,ylo=ylo,yhi=yhi)
#        figname="fig"+"%03d" % n+".png"
#        fign.savefig(figname)
#        plt.close(fign)
#        fignamelist.append(figname)
#    makemontage(fignamelist,nrows,montagetitle)
#    if(removefigs):
#        deletefiglist(fignamelist)

def indxstr(n,indxlist):
    if(indxlist==None):
        return str(n)
    else:
        return str(int(indxlist[n]))

def indxrange(indxlist,nmin=None,nmax=None):
    retnmin=0
    retnmax=len(indxlist)
    if(indxlist!=None):
        if(nmin!=None):
            retnmin=list(indxlist).index(nmin)
        if(nmax!=None):
            retnmax=list(indxlist).index(nmax)+1
    print("retnmin,retnmax\t"+str(retnmin)+", "+str(retnmax))
    return retnmin,retnmax

def makemontage(fignamelist,nmin=None,nmax=None,nrows=None,outfigname="montage.png"):
    nfigs=len(fignamelist)
    fignamestr=" ".join(fignamelist)
    if(nrows==None):
        nrows=int(floor(sqrt(nfigs)))
    ncols=int(ceil(nfigs/nrows))
    montagecommand="montage -tile "+str(nrows)+"x"+str(ncols)+ " -mode Concatenate "+fignamestr+" "+outfigname
    subprocess.call(montagecommand.split())

def deletefiglist(fignamelist):
    #print("fignamelist\t"+str(fignamelist))
    for figname in fignamelist:
        os.remove(figname)


def vecstoarrays(vec1,vec2):
    n=len(vec1)
    m=len(vec2)
    retarray1=outer(vec1,ones(m))
    retarray2=outer(ones(n),vec2)
    return retarray1,retarray2

################################################################################ 
#Deconvolution

def sinsqpulse(w0,wenv,tarray):
    tmax=pi/wenv
    retarray=zeros(len(tarray))*0j
    for i in range(len(tarray)):
        if(tarray[i]<=tmax):
            retarray[i]=exp(1j*w0*tarray[i])*pow(sin(wenv*tarray[i]),2)
    return retarray

def pulseenvelopepower(wenv,tarray,power):
    tmax=pi/wenv
    retarray=zeros(len(tarray))*0j
    for i in range(len(tarray)):
        if(tarray[i]<=tmax):
            retarray[i]=pow(sin(wenv*tarray[i]),2*abs(power))
    return retarray

def deconvolve_padded(inpvec,convvec):
    ninp=len(inpvec)
    nconv=len(convvec)
    padinpvec=zeros(len(inpvec)+len(convvec))*0j
    padinpvec[nconv/2:nconv/2+ninp]=inpvec[:]
    deconvvec=deconvolve(padinpvec,convvec)
    return deconvvec

def fouriermat(inpvec):
    nvec=len(inpvec)
    retmat=zeros((nvec,nvec))*0j
    for j in range(nvec):
        retmat[:,j]=roll(inpvec,-int(floor(nvec/2))+j)
#    for i,j in product(range(nvec),range(nvec)):
##        k=i-j
##        if((k>=-nvec/2) and (k<nvec/2)):
##            retmat[i,j]=inpvec[nvec/2+k]
#        retmat[mod(i-nvec/2+j,nvec),j]=inpvec[i]
    return retmat

def pulseftmat(wnlist,wenv,tarray):
    nt=len(tarray)
    retarray=identity(nt)*(1.+0j)
    for i in range(len(wnlist)):
        [w0,n]=wnlist[i]
        if(n>=0):
            pulsevec=sinsqpulse(w0,wenv,tarray)
        else:
            pulsevec=sinsqpulse(-w0,wenv,tarray)
        ftpulsevec=normvec(fftshift(fft(pulsevec)))
        ftpulsemat=fouriermat(ftpulsevec)
        retarray=dot(matrix_power(ftpulsemat,abs(n)),retarray)
    return retarray

def normvec(inpvec):
    norm=0.
    for i in range(len(inpvec)):
        norm+=abs(inpvec[i])
    return inpvec/sqrt(norm)

def hcpulseftmat(wnlist,wenv,tarray):
    return conjugate(transpose(pulseftmat(wnlist,wenv,tarray)))

def removepulsespectrum(i0,i1,ncycles,sourcearray,ws=.0565):
    wpr=winpvec[i0]
    nk=list(kIRvec).index(i1)#int(kIRvec[i1])
    print("nk\t"+str(nk))
    wnlist=[[wpr,1],[ws,i1]]
    pftmat=pulseftmat(wnlist,ws/(2*ncycles),tvec)
    retvec=solve(pftmat,sourcearray[i0,i1,:])
    return retvec

def removepulsespectrum_array(inparray,ncycles=8,ws=.0565):
    (n0,n1,n2)=shape(inparray)
    retarray=zeros(shape(inparray))*0j
    for i,j in product(range(n0),range(n1)):
        retarray[i,j,:]=removepulsespectrum(i,int(kIRvec[j]),ncycles,inparray,ws)
    return retarray

def removepulsespectrum_array2(ncycles=8,ws=.0565):
    wenv=ws/(2*ncycles)
    wktnofreqarray=wkt_subtracttwofreqs()
    wkwretarray=zeros(shape(wktnofreqarray))*0j
    (n0,n1,n2)=shape(wktnofreqarray)
    for i,j in product(range(n0),range(n1)):
        print("i,j\t"+str(i)+"\t"+str(j))
        power=1+abs(j-int(floor(n1/2)))
        wkwretarray[i,j,:]=deconvolvepulse(wktnofreqarray[i,j,:],power,wenv)
    return wkwretarray

def deconvolvepulse(inpvec,power,wenv):
    ftinpvec=ftarray(inpvec,axes=[0])
    envvec=pulseenvelopepower(wenv,tvec,power)
    ftenvvec=ftarray(envvec,axes=[0])
    ftenvmat=fouriermat(ftenvvec)
    retvec=solve(ftenvmat,ftinpvec)
    return retvec

def dt_evolution_array(inparray):
    retarray=copy(inparray)
    (n0,n1,n2,n3)=shape(retarray)
    for i in range(n0):
        retarray[i,:,:,:]-=inparray[0,:,:,:]
    return retarray


def dt_contourplot(xvec, yvec, dt_npr_ns_yarray, nprval=1, nsval=0,
                       **kwargs):
    xarray,yarray=vecstoarrays(xvec,yvec)
    tmpzarray=dt_npr_ns_yarray[:, list(nprvec).index(nprval),
                               list(nsvec).index(nsval), :]
    return contourplot(xarray,yarray,tmpzarray, **kwargs)
