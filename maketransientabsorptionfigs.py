import sys
sys.path.append(r'/Users/zwalters/pythonscripts')
from makefigs import *
import subprocess

def nm_to_eV(nm):
    hc=1239.841#planck's constant in eV*nm
    eV=hc/nm
    return eV

def transientabsorption(diparray,Earray):
    (n0,n1)=shape(diparray)
    retarray=zeros((n0,n1))
    for i,j in product(range(n0), range(n1)):
        retarray[i,j]=2*imag(diparray[i,j] * conjugate(Earray[i,j])) /pow(abs(Earray[i,j]),2)
    return retarray

def transientabsorption2(diparray,Evec):
    (n0,n1)=shape(diparray)
    retarray=zeros((n0,n1))
    for i in range(n0):
        retarray[i,:]=2*imag(diparray[i,:]* conjugate(Evec[:])) #/ pow(abs(Evec[:]),2)
    return retarray

#remove streaking frequencies
def transient_absorption_figure(ndtplot,ws, wpr, nophase=False, zeropad=False,
                                nspad=101, **kwargs):
        
    ntarray=nsvstarray[ndtplot,nprvec.index(1),:,:]
    if(nophase):
        for i,j in product(range(nKs),range(nt)):
            ntarray[i,j]*=exp(-1j*ws*nsvec[i]*tmeasurevec[j])#-1j*wpr*tmeasurevec[j])


    #fourier transform
    phiwarray=nDFFT(ntarray,axislist=[1],shiftaxes=True)
    phiwarray=nDFFT(phiwarray,axislist=[0],shiftaxes=True,inv=True)
    phiarray,warray=vecstoarrays(IRCEPvec,woutvec)

    #transient absorption
    Evec=phaseEwarray[ndtplot,0,0,:]
    transarray=transientabsorption2(phiwarray,Evec)
    if(not(zeropad)):
        phiticks=[0,.5,1,1.5,2]
        retfig=imshowplot(phiarray,warray,transarray, ymultfactor=Hrt, ylo=evlo,
                          yhi=evhi, symmetricrange=True, xticks=phiticks, **kwargs)
    else:
        #zero pad
        nspadstart=-int(floor(nspad/2))
        padnsvec=list(range(nspadstart,nspadstart+nspad))
        ntarray_nophase_padded=zeros((nspad,nt))*0j
        ntarray_padded=zeros((nspad,nt))*0j
        for n in nsvec:
            indx1=list(nsvec).index(n)
            indx2=list(padnsvec).index(n)
            ntarray_padded[indx2,:]=ntarray[indx1,:]

        phiwarray_padded=nDFFT(ntarray_padded,axislist=[1],shiftaxes=True)
        phiwarray_padded=nDFFT(phiwarray_padded,axislist=[0],shiftaxes=True,inv=True)

        padphivec=arange(0,2,2./nspad)
        phiarray,warray=vecstoarrays(padphivec,woutvec)

        transarray_padded=transientabsorption2(phiwarray_padded,Evec)
        phiticks=[0,.5,1,1.5,2]
        retfig=imshowplot(phiarray,warray,transarray_padded, ymultfactor=Hrt, ylo=evlo,
                        yhi=evhi, symmetricrange=True, xticks=phiticks, **kwargs)
    return retfig


evlo=20
evhi=30
nlo=None
nhi=None
savefigs=True
ndtplot=0#100

#Load pickled arrays
nsvec=load("kIRvec.pkl")
woutvec=load("woutvec.pkl")
IRCEPvec=load("IRcepvec.pkl")
deltatvec=load("deltatvec.pkl")
tmeasurevec=load("tmeasurevec.pkl")

nsvstarray=load("kIRvstarray.pkl")
nsvswarray=load("kIRvswarray.pkl")
phaseEwarray=load("phaseEwarray.pkl")

#get shapes
(ndeltat,nKpr,nKs, nt)=shape(nsvstarray)

nprstart=-int(floor(nKpr/2))
nprvec=list(range(nprstart,nprstart+nKpr))

##plot ns_vs_t
#ntarray=nsvstarray[ndtplot,nprvec.index(1),:,:]
#nsarray,tarray=vecstoarrays(nsvec,tmeasurevec)
##fig0d=contourplot(nsarray,tarray,abs(ntarray), ncontours=14, xlo=nlo, xhi=nhi,
##                  ymultfactor=aut/1000, xlabel="$n_{s}$", ylabel="$t-t_{0}$ (fs)",
##                  ylo=0, yhi=30)
#
##plot phi_vs_t
#phitarray=nDFFT(ntarray, axislist=[0], shiftaxes=True, inv=True)
#phiarray,tarray=vecstoarrays(IRCEPvec,tmeasurevec)
##fig0c=imshowplot(phiarray,tarray,abs(phitarray), xlabel="$\phi/\pi$", 
##                 ylabel="$t-t_{0}$ (fs)", ymultfactor=aut/1000, ylo=0, yhi=30,
##                 xlo=nlo, xhi=nhi, cmap="cubehelix_r", symmetricrange=False)
#
##plot phi_vs_w
#phiwarray=nDFFT(phitarray, axislist=[1], shiftaxes=True)
#phiarray, warray=vecstoarrays(IRCEPvec, woutvec)
##fig0b=imshowplot(phiarray,warray,abs(phiwarray), xlabel="$\phi/\pi$", 
##                 ylabel="$\omega_{out}$ (eV)", ymultfactor=Hrt, ylo=20, yhi=28, 
##                 cmap="cubehelix_r", symmetricrange=False)
#
##plot ns_vs_w
#nwarray=nDFFT(ntarray,axislist=[1], shiftaxes=True)
#nsarray,warray=vecstoarrays(nsvec,woutvec)
##fig0a=contourplot(nsarray,warray,abs(nwarray),ncontours=14, ymultfactor=Hrt,
##                  ylo=15, yhi=35, xlo=nlo, xhi=nhi, xlabel="$n_{s}$", ylabel = 
##                  "$\omega_{out}$ (eV)")
#

minzval=-.15
maxzval=.15
ws=nm_to_eV(760)/Hrt
wpr=25/Hrt
gammaval=1

tmpcmap=diverginghsv()#divergingcubehelix()

fignamelist=[]
for ndtplot in range(0,len(deltatvec),1):
    legendstr="$t_{xuv}-t_{IR}=$"+str(round(deltatvec[ndtplot]*aut/1000,2))+" (fs)"
    #legendstr="$t_{xuv}-t_{IR}=$"+str(deltatvec[ndtplot]*aut/1000)+" (fs)"
    fign=transient_absorption_figure(nophase=True, ndtplot=ndtplot, ws=ws,
                                     wpr=wpr, zeropad=True, nspad=100,
                                     colorpower=1,
                                     gamma=gammaval,#legend=legendstr,
                                     xlabel="$\phi/\pi$",
                                     ylabel="$\omega_{out}$ (eV)",
                                     minzval=minzval, maxzval=maxzval,
                                     inpfontsize=30, cmap=tmpcmap)
    figname="fig"+"%03d" % ndtplot+".png"
    fign.savefig(figname, bbox_inches="tight")
    plt.close(fign)
    fignamelist.append(figname)
fignamestr=" ".join(fignamelist)
movietitle="movie.mp4"
#subprocess.call(["ffmpeg", "-r", "3",
#                 "-i",fignamelist,"-pix_fmt", "yuv420p",movietitle])
#commandstr="ffmpeg -r 5 -i %*.png -pix_fmt yuv420p "+movietitle
commandstr="ffmpeg -framerate 10 -i fig%03d.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 23 -pix_fmt yuv420p -r 30 "+movietitle
subprocess.call(commandstr, shell=True)
print("command string\n"+commandstr)
print("created "+movietitle)
plt.show()
