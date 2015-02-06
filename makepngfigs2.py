import sys
sys.path.append(r'/Users/zwalters/pythonscripts')
from makefigs import *

def nm_to_eV(nm):
    hc=1239.841#planck's constant in eV*nm
    eV=hc/nm
    return eV

evlo=15#22
evhi=35#28
nlo=-20
nhi=20
savefigs=True

#Load pickled arrays
nsvec=load("kIRvec.pkl")
woutvec=load("woutvec.pkl")
IRCEPvec=load("IRcepvec.pkl")
deltatvec=load("deltatvec.pkl")
tmeasurevec=load("tmeasurevec.pkl")

nsvstarray=load("kIRvstarray.pkl")
nsvswarray=load("kIRvswarray.pkl")

#get shapes
(ndeltat,nKpr,nKs, nt)=shape(nsvstarray)

nprstart=-int(floor(nKpr/2))
nprvec=list(range(nprstart,nprstart+nKpr))

#hsv colormap
hsvcmap=hsvhelixcmap(thetastart=0,nturns=1, reverse=False)

#as a guide to temporal scale, make ticks every 100 attoseconds
dtick=100/aut
w0=nm_to_eV(800)/Hrt
period=2*pi/w0
topxticks=arange(0,period,dtick)/period*2

#plot ns_vs_t
ntarray=nsvstarray[0,nprvec.index(1),:,:]
nsarray,tarray=vecstoarrays(nsvec,tmeasurevec)
fig0d=contourplot(nsarray,tarray,abs(ntarray), ncontours=14, xlo=nlo, xhi=nhi,
                  ymultfactor=aut/1000, xlabel="$n_{s}$", ylabel="$t-t_{0}$ (fs)",
                  ylo=0, yhi=30, contourcmap=hsvcmap)

#plot phi_vs_t
phitarray=nDFFT(ntarray, axislist=[0], shiftaxes=True, inv=True)
phiarray,tarray=vecstoarrays(IRCEPvec,tmeasurevec)
fig0c=imshowplot(phiarray,tarray,abs(phitarray), xlabel="$\phi/\pi$", 
                 ylabel="$t-t_{0}$ (fs)", ymultfactor=aut/1000, ylo=0, yhi=30,
                 cmap=hsvcmap, symmetricrange=False)

#plot phi_vs_w
phiwarray=nDFFT(phitarray, axislist=[1], shiftaxes=True)
nsarray, warray=vecstoarrays(IRCEPvec, woutvec)
fig0b=imshowplot(nsarray,warray,abs(phiwarray), xlabel="$\phi/\pi$", 
                 ylabel="$\omega_{out}$ (eV)", ymultfactor=Hrt, ylo=evlo, yhi=evhi, 
                 cmap=hsvcmap, symmetricrange=False)

#plot ns_vs_w
nwarray=nDFFT(ntarray,axislist=[1], shiftaxes=True)
nsarray,warray=vecstoarrays(nsvec,woutvec)
fig0a=contourplot(nsarray,warray,abs(nwarray),ncontours=14, ymultfactor=Hrt,
                  ylo=0, yhi=50, xlo=nlo, xhi=nhi, xlabel="$n_{s}$", ylabel = 
                  "$\omega_{out}$ (eV)", contourcmap=hsvcmap)


#remove streaking frequencies
ws=nm_to_eV(800)/Hrt
wpr=25/Hrt
evlo_nofreq=evlo
evhi_nofreq=evhi
ntarray_nophase=copy(ntarray)
for i,j in product(range(nKs),range(nt)):
    ntarray_nophase[i,j]*=exp(-1j*ws*nsvec[i]*tmeasurevec[j])#-1j*wpr*tmeasurevec[j])

fig1p=imshowplot_hsv(nsarray,tarray,ntarray_nophase,colorpower='log', xlo=nlo, xhi=nhi)

#fourier transform
phiwarray=nDFFT(ntarray_nophase,axislist=[1],shiftaxes=True)
phiwarray=nDFFT(phiwarray,axislist=[0],shiftaxes=True,inv=True)
phiarray,warray=vecstoarrays(IRCEPvec,woutvec)
fig2p=imshowplot(phiarray,warray,abs(phiwarray), colorpower="log",
                 cmap=hsvcmap, symmetricrange=False, ymultfactor=Hrt,
                 ylo=evlo_nofreq, yhi=evhi_nofreq, gamma=1., topxticks=topxticks)
fig2pp=imshowplot_hsv(phiarray,warray,phiwarray, colorpower=1.,ymultfactor=Hrt,
                      ylo=evlo_nofreq, yhi=evhi_nofreq)

#zero pad
nspad=101
nspadstart=-int(floor(nspad/2))
padnsvec=list(range(nspadstart,nspadstart+nspad))
ntarray_nophase_padded=zeros((nspad,nt))*0j
for n in nsvec:
    indx1=list(nsvec).index(n)
    indx2=list(padnsvec).index(n)
    ntarray_nophase_padded[indx2,:]=ntarray_nophase[indx1,:]
phiwarray_padded=nDFFT(ntarray_nophase_padded,axislist=[1],shiftaxes=True)
phiwarray_padded=nDFFT(phiwarray_padded,axislist=[0],shiftaxes=True,inv=True)
padphivec=arange(0,2,2./nspad)
phiarray,warray=vecstoarrays(padphivec,woutvec)



fig3p=imshowplot(phiarray,warray,abs(phiwarray_padded), colorpower="log",
                 ymultfactor=Hrt, ylo=evlo_nofreq, yhi=evhi_nofreq, symmetricrange=False, gamma=1., cmap=hsvcmap, topxticks=topxticks)
fig3pp=imshowplot_hsv(phiarray,warray,phiwarray_padded, colorpower="log",
                 ymultfactor=Hrt, ylo=evlo_nofreq, yhi=evhi_nofreq)

if(savefigs):
    prefix="wpr_25_eV_ws_800_nm_I_3em3_"
    fig0c.savefig(prefix+"phi_vs_t.png", bbox_inches="tight")
    fig0a.savefig(prefix+"ns_vs_w.png", bbox_inches="tight")
    fig0b.savefig(prefix+"phi_vs_w.png", bbox_inches="tight")
    fig0d.savefig(prefix+"ns_vs_t.png", bbox_inches="tight")
    fig2p.savefig(prefix+"phi_vs_w_nostreakingphase.png", bbox_inches="tight")
    fig3p.savefig(prefix+"phi_vs_w_nostreakingphase_zeropadded.png",
                  bbox_inches="tight")

plt.show()
