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
wphiwarray=[]
wphiwarray=load("freq_vs_phase_vs_freq.pkl")
wphitarray=[]
wphitarray=load("freq_vs_phase_vs_t.pkl")
wkIRwarray=[]
wkIRwarray=load("freq_vs_kIR_vs_freq.pkl")
wkIRtarray=[]
wkIRtarray=load("freq_vs_kIR_vs_t.pkl")
kIRvec=[]
kIRvec=load("kIRvec.pkl")
phasevec=[]
phasevec=load("phasevec.pkl")
winpvec=[]
winpvec=load("winpvec.pkl")
woutvec=[]
woutvec=load("woutvec.pkl")
tvec=[]
tvec=load("tvec.pkl")


####################################################################
#processed arrays
def wkt_subtracttwofreqs(wosc=.0565):
    retarray=copy(wkIRtarray)
    (n0,n1,n2)=shape(retarray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(-1j*(winpvec[i]+kIRvec[j]*wosc)*tvec[k])
    return retarray


def wkt_subtractprobefreqs():
    retarray=copy(wkIRtarray)
    (n0,n1,n2)=shape(retarray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(-1j*(winpvec[i])*tvec[k])
    return retarray

def wkw_subtractprobefreqs():
    dw_kIR_tarray=wkt_subtractprobefreqs()#subtract probe frequencies
    dw_kIR_warray=fftshift(fftn(dw_kIR_tarray,axes=[2]),axes=[2])
    return dw_kIR_warray

def dwphiw_subtractprobefreqs(wosc=.0565):
    dw_kIR_warray=wkw_subtractprobefreqs()
    dw_phase_warray=fftshift(ifftn(dw_kIR_warray,axes=[1]),axes=[1])
    (n0,n1,n2)=shape(dw_phase_warray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        dw_phase_warray[i,j,k]*=exp(-1j*(winpvec[i])*phasevec[j]/wosc)
    return dw_phase_warray
    

def dwkw_subtracttwofreqs(wosc=.0565):
    retarray=wkt_subtracttwofreqs(wosc)
    retarray=fftn(retarray,axes=[2])
    retarray=fftshift(retarray,axes=[2])
    return retarray

def dwphiw_subtracttwofreqs(wosc=.0565):
    retarray=dwkw_subtracttwofreqs(wosc)
    retarray=iftarray(retarray,axes=[1])
    return retarray

################################################################################
#array operations

def iftarray(inparray,axes,shift=True):
    if(shift):
        retarray=fftshift(inparray,axes=axes)
    else:
        retarray=copy(inparray)
    retarray=ifftn(retarray,axes=axes)
    return retarray

def subtractdwphase(inparray,wosc=.0565):
    retarray=copy(inparray)
    (n0,n1,n2)=shape(inparray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(1j*phasevec[j]/wosc*woutvec[j])
    return retarray


def subtractdrivingfreqs(wosc=.0565):
    retarray=copy(wkIRtarray)
    (n0,n1,n2)=shape(retarray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(-1j*(winpvec[i]+kIRvec[j]*wosc)*tvec[k])
    return retarray

def subtractprobefreqs():
    retarray=copy(wkIRtarray)
    (n0,n1,n2)=shape(retarray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(-1j*(winpvec[i])*tvec[k])
    return retarray

def deltaw_vs_w(wosc=.0565):
    retarray=subtractdrivingfreqs(wosc)
    retarray=fftn(retarray,axes=[2])
    retarray=fftshift(retarray,axes=[2])
    return retarray

def iftarray(inparray,axes,shift=True):
    if(shift):
        retarray=fftshift(inparray,axes=axes)
    else:
        retarray=copy(inparray)
    retarray=ifftn(retarray,axes=axes)
    return retarray

def subtractdwphase(inparray,wosc=.0565):
    retarray=copy(inparray)
    (n0,n1,n2)=shape(inparray)
    for i,j,k in product(range(n0),range(n1),range(n2)):
        retarray[i,j,k]*=exp(1j*phasevec[j]/wosc*woutvec[j])
    return retarray
    

def ftarray(inparray,axes,shift=True):
    retarray=fftn(inparray,axes=axes)
    if(shift):
        retarray=fftshift(retarray,axes=axes)
    return retarray

def wphiw_phasecorrected(wosc=.0565):
    wkIRwarray

def zeroinitialphase(inparray):
    anglearray=angle(inparray[:,0,:])
    retarray=copy(inparray)
    (n0,n1,n2)=shape(inparray)
    for i in range(n1):
        retarray[:,i,:]*=exp(-1j*anglearray)
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
def arraymovie(inparray, movietitle="movie.mp4", colorpower='log', logrange=6,
               arrayfun=None, xvec=winpvec, yvec=woutvec,
               removefigs=True, theme='hls', ylo=None, yhi=None):
    (n0,n1,n2)=shape(wkIRwarray)
    print("shape wkIRwarray\t"+str(shape(wkIRwarray)))
    xarray,yarray=vecstoarrays(xvec,yvec)

    zmax=(abs(inparray)).max()
    fignamelist=[]
    for n in range(shape(inparray)[1]):
        if(arrayfun==None):
            fign=imshowplot_hsv(xarray, yarray, inparray[:, n, :],
                                colorpower=colorpower, logrange=logrange,
                                legend="$\phi=$"+"{0:.2f}".format(phasevec[n])+"$\pi$",
                                ymultfactor=Hrt, xmultfactor=Hrt,
                                absmax=zmax,  xlabel="$\omega_{in}$ (eV)",
                                ylabel="$\omega_{out}$ (eV)",  theme=theme,
                                ylo=ylo,  yhi=yhi)
        else:
            fign=imshowplot_fun(xarray, yarray, inparray[:, n, :],
                                colorpower=colorpower, logrange=logrange,
                                legend="$\phi=$"+"{0:.2f}".format(phasevec[n])+"$\pi$",
                                ymultfactor=Hrt, xmultfactor=Hrt,
                                arrayfunction=arrayfun, absmax=zmax,
                                xlabel="$\omega_{in}$ (eV)",
                                ylabel="$\omega_{out}$ (eV)", theme=theme,
                                ylo=ylo,  yhi=yhi)
        figname="fig"+"%03d" % n+".png"
        fign.savefig(figname)
        plt.close(fign)
        fignamelist.append(figname)
#        fign=interpplot(winparray.flatten(),woutarray.flatten(),
#                   real(iftnoconstarray[:,n,:].flatten()),
#                        legend="$\phi=$"+"{0:.2f}".format(phasevec[n])+"$\pi$",
#                        colorpower=1,
#                   inpcmapname='RdBu',ylo=0,ymultfactor=Hrt,
#                   xmultfactor=Hrt,zmax=zmax)
    print(" ".join(fignamelist))
    subprocess.call(["ffmpeg", "-r", "3",
                     "-i","fig%03d.png","-pix_fmt", "yuv420p",movietitle])
    print("created "+movietitle)
    if(removefigs):
        figfiles=glob.glob("fig*.png")
        for figfile in figfiles:
            os.remove(figfile)

##variants of array movie
#def removeftindicesmovie(ftarray,colnumlist=[],movietitle="movie.mp4",colorpower='log',logrange=6,arrayfun=None,
#                         tmpwinpvec=winpvec,tmpwoutvec=woutvec):
#    (n0,n1,n2)=shape(ftarray)
#    indxlist=ftindexlist(n1,colnumlist)
#    tmparray=iftindx1(zeroarraycolumns(ftarray,indxlist))
#    arraymovie(tmparray,movietitle=movietitle,colorpower=colorpower,logrange=logrange,arrayfun=arrayfun,
#               tmpwinpvec=tmpwinpvec,tmpwoutvec=tmpwoutvec)
#
#def subtractmeanarraymovie(inparray, movietitle="movie.mp4", colorpower='log',
#                          logrange=6, arrayfun=None):
#    (n0,n1,n2)=shape(ftarray)
#    tmparray=subtractarraymean(inparray)
#    arraymovie(tmparray, movietitle=movietitle, colorpower=colorpower,
#               logrange=logrange, arrayfun=arrayfun)
#
#def zerooriginalarrayphasemovie(inparray, movietitle="movie.mp4", colorpower='log',
#                          logrange=6, arrayfun=None):
#    tmparray=zerooriginalarrayphase(inparray)
#    arraymovie(tmparray, movietitle=movietitle, colorpower=colorpower,
#               logrange=logrange, arrayfun=arrayfun)
#
##    subprocess.call(["convert","-quality", "95", 
##                         " ".join(fignamelist),"diffmovie.mpeg"])
#def dipole_vs_phase_movie(movietitle="dipole_vs_phase.mp4",colorpower='log',
#                          logrange=6,arrayfun=None,theme='hls'):
#    tmpwphiwarray=load("freq_vs_phase_vs_freq.pkl")
#    tmpkIRvec=load("kIRvec.pkl")
#    tmpphasevec=load("phasevec.pkl")
#    tmpwinpvec=load("winpvec.pkl")
#    tmpwoutvec=load("woutvec.pkl")
#    arraymovie(tmpwphiwarray, movietitle=movietitle, colorpower=colorpower,
#               logrange=logrange, arrayfun=arrayfun, tmpwinpvec=tmpwinpvec,
#               tmpwoutvec=tmpwoutvec, theme=theme)
#
#def dipole_vs_phase_movie_noconst(movietitle="dipole_vs_phase_noconst.mp4",
#                                  colorpower='log', logrange=6, arrayfun=None):
#    tmpwkIRwarray=load("freq_vs_kIR_vs_freq.pkl")
#    tmpwinpvec=load("winpvec.pkl")
#    tmpwoutvec=load("woutvec.pkl")
#    removeftindicesmovie(tmpwkIRwarray,colnumlist=[0],movietitle=movietitle,colorpower=colorpower,logrange=logrange,
#                         arrayfun=arrayfun,tmpwinpvec=tmpwinpvec,tmpwoutvec=tmpwoutvec)
#
#def abs_minus_const_movie(movietitle="abs_minus_const.mp4",
#                                  colorpower='log',logrange=6,arrayfun=None):
#    tmpwphiwarray=load("freq_vs_phase_vs_freq.pkl")
#    tmpkIRvec=load("kIRvec.pkl")
#    tmpphasevec=load("phasevec.pkl")
#    tmpwinpvec=load("winpvec.pkl")
#    tmpwoutvec=load("woutvec.pkl")
#    tmparray=subtractarraymean(abs(tmpwphiwarray))
#    arraymovie(tmparray,movietitle=movietitle,colorpower=colorpower,
#               logrange=logrange,arrayfun=arrayfun,tmpwinpvec=tmpwinpvec,
#               tmpwoutvec=tmpwoutvec)
#
#def makemovies(colorpower='log',logrange=6,theme='hls'):
#    dipole_vs_phase_movie(movietitle="dipole_vs_phase.mp4",colorpower=colorpower,
#                          logrange=logrange,theme=theme)
##    dipole_vs_phase_movie(movietitle="abs_dipole_vs_phase.mp4",colorpower=colorpower,logrange=logrange,arrayfun=logz)
##    dipole_vs_phase_movie_noconst(movietitle="dipole_vs_phase_noconst.mp4",colorpower=colorpower,logrange=logrange)
##    dipole_vs_phase_movie_noconst(movietitle="abs_dipole_vs_phase_noconst.mp4",colorpower=colorpower,logrange=logrange,arrayfun=logz)
##    abs_minus_const_movie(movietitle="abs_minus_const.mp4",
##                          colorpower=colorpower,logrange=logrange)
############################################################################
##makemovies(logrange=log(1e4),theme='hls')





####################################################################
#Montage tools
def arraymontage(inparray, indxlist=kIRvec, montagetitle="montage.png",
                 colorpower='log', logrange=6, arrayfun=None,
                 xvec=winpvec, yvec=woutvec, removefigs=True,
                 nrows=None, nmin=None, nmax=None, nstride=1,  ylo=None, yhi=None,
                 xlabel="$\omega_{in}$ (eV)", ylabel="$\omega_{out}$ (eV)",
                 arraynormalize=True):
    (n0,n1,n2)=shape(inparray)
    if(arraynormalize):
        zmax=abs(inparray).max()
    else:
        zmax=None
    fignamelist=[]

    xarray,yarray=vecstoarrays(xvec,yvec)

    loopnmin,loopnmax=indxrange(indxlist,nmin,nmax)
    for n in range(loopnmin,loopnmax,nstride):
        if(arrayfun==None):
            fign=imshowplot_hsv(xarray,yarray,inparray[:,n,:],
                                colorpower=colorpower,logrange=logrange,
                                legend="$n_{streak}=$"+indxstr(n,indxlist),
                                ymultfactor=Hrt, xmultfactor=Hrt, absmax=zmax,
                                xlabel=xlabel, ylabel=ylabel,ylo=ylo,yhi=yhi)
        else:
            fign=imshowplot_fun(xarray,yarray,inparray[:,n,:],
                                colorpower=colorpower,logrange=logrange,
                                legend="$n_{streak}=$"+indxstr(n,indxlist),
                                ymultfactor=Hrt, xmultfactor=Hrt,
                                arrayfunction=arrayfun, absmax=zmax,
                                xlabel=xlabel, ylabel=ylabel,ylo=ylo,yhi=yhi)
        figname="fig"+"%03d" % n+".png"
        fign.savefig(figname)
        plt.close(fign)
        fignamelist.append(figname)
    makemontage(fignamelist,nrows,montagetitle)
    if(removefigs):
        deletefiglist(fignamelist)

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

