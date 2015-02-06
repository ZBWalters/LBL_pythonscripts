import sys
import matplotlib
sys.path.append(r'/Users/zwalters/pythonscripts')
from makefigs import *
from matplotlib import cm
from matplotlib._cm import cubehelix
from itertools import product as iterproduct

def nm_to_eV(nm):
    hc=1239.841#planck's constant in eV*nm
    eV=hc/nm
    return eV

def greaterindx(inplist,gtval):
    return next(x[0] for x in enumerate(inplist) if x[1]>gtval)

def indxrange(vec,loval,hival):
    loindx=greaterindx(vec,loval)
    hiindx=greaterindx(vec,hival)
    return loindx,hiindx

def maskarray(xarray,yarray,zarray,xlo=None,xhi=None,ylo=None,yhi=None,xmultfactor=1.,ymultfactor=1.):
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
#    print("shape xarray\t"+str(shape(xarray)))
#    print("shape yarray\t"+str(shape(yarray)))
#    print("shape zarray\t"+str(shape(zarray)))
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

def printarrays(xarray,yarray,zarray,outfile="arrays.txt"):
    (n0,n1)=shape(xarray)
    f=open(outfile,'w')
    for i in range(n0):
        for j in range(n1):
            f.write(str(xarray[i,j])+"\t"+str(real(yarray[i,j]))+"\t"+str(zarray[i,j])+"\n")
        f.write("\n")
    f.close


def transientabsorption(diparray,Earray):
    (n0,n1)=shape(diparray)
    retarray=zeros((n0,n1))
    for i,j in product(range(n0), range(n1)):
        retarray[i,j]=2*imag(diparray[i,j] * conjugate(Earray[i,j])) #/pow(abs(Earray[i,j]),2)
    return retarray
#
#def combinefuncs(x,f1,f2, rnge=.8):
#    if(x>.5):
#        return f2((1.-2*(x-.5))*rnge+(1-rnge))
#    else:
#        return f1(2*x*rnge+(1-rnge))
#
#def cdictfuncstosegments(cdict1,cdict2,nsegments=101, rnge=1.):
#    dx=1./nsegments
#    xarray=arange(0,1+dx,dx)
#    redlist=[]
#    bluelist=[]
#    greenlist=[]
#    red1=cdict1['red']
#    red2=cdict2['red']
#    blue1=cdict1['blue']
#    blue2=cdict2['blue']
#    green1=cdict1['green']
#    green2=cdict2['green']
#    
#    for i in range(len(xarray)):
#        redval=combinefuncs(xarray[i],red1,red2,rnge=rnge)
#        redval=min(redval,1.)
#        redval=max(redval,0.)
#        blueval=combinefuncs(xarray[i],blue1,blue2,rnge=rnge)
#        blueval=min(blueval,1.)
#        blueval=max(blueval,0.)
#        greenval=combinefuncs(xarray[i],green1,green2,rnge=rnge)
#        greenval=min(greenval,1.)
#        greenval=max(greenval,0.)
#        redlist.append((xarray[i],redval,redval))
#        bluelist.append((xarray[i],blueval,blueval))
#        greenlist.append((xarray[i],greenval,greenval))
##    print("redlist\t"+str(redlist))
#    retcdict={'red':redlist, 'green':greenlist, 'blue':bluelist}
#    return retcdict
#
#
#
#
#def divergingcubehelix(s1=0.6, r1=-0.5, s2=0, r2=0.5, hue=2., gamma=1.,nsegments=101,rnge=1.):
#    #tmpcdict1=cubehelix(gamma=gamma,s=2,r=rot,h=hue)
#    #tmpcdict2=cubehelix(gamma=gamma,s=1,r=-rot,h=hue)
#    tmpcdict1=cubehelix(gamma=gamma, s=s1, r=r1, h=hue)
#    tmpcdict2=cubehelix(gamma=gamma, s=s2, r=r2, h=hue)
#    tmpcdict3list=cdictfuncstosegments(cdict1=tmpcdict1, cdict2=tmpcdict2,
#                                       nsegments=nsegments, rnge=rnge)
#    tmpcmap=matplotlib.colors.LinearSegmentedColormap("divergingcubehelix",tmpcdict3list)
#    return tmpcmap
#
################################################################################
contourcmapname='coolwarm'#'redblue'
densitycontourcmapname='gray'

phasevswarray=load("phasevswarray.pkl")
kIRvswarray=load("kIRvswarray.pkl")
kIRvstarray=load("kIRvstarray.pkl")
phaseEwarray=load("phaseEwarray.pkl")
nsvec=list(load("kIRvec.pkl"))
tmeasurevec=list(load("tmeasurevec.pkl"))
dtvec=list(load("deltatvec.pkl"))
woutvec=list(load("woutvec.pkl"))
npr=shape(kIRvswarray)[1]
nprlo=-int(floor(npr/2))
nprvec=list(range(nprlo,nprlo+npr))

##test: keep same pulse as a function of dt
#(n0,n1,n2,n3)=shape(phaseEwarray)
#for i in range(1,n0):
#    phaseEwarray[i,:,:,:]=phaseEwarray[0,:,:,:]
##end test

dtarray,warray=vecstoarrays(dtvec,woutvec)
(ndt,npr,ns,nw)=shape(kIRvswarray)
absorbarray=zeros((ndt,ns,nw))

#make plot of transient absorption without phase matching
evlo=20#20
evhi=26#26
suffix="_20eV_to_26eV_hsvhelix"
gammaval=1.#1.#5.
savefigs=False
inpfontsize=30
minzval=-.05#None#-.03#None#-.05
maxzval=.05#None#.03#None#.05
absorbarray0=transientabsorption(phasevswarray[:,0,0,:], phaseEwarray[:,0,0,:])#2*imag(phasevswarray[:,0,0,:]*conjugate(phaseEwarray[:,0,0,:])/pow(abs(phaseEwarray[:,0,0,:]),2))

rot=0
#tmpcmap=divergingcubehelix(gamma=gammaval, rnge=.9, starthue1=60, endhue1=180,
#                           starthue2=60, endhue2=-60)
tmpcmap=diverginghsv()
#tmpcmap=hsvhelixcmap(thetastart=0,nturns=-1)


fign=imshowplot(dtarray, warray, absorbarray0,
                ymultfactor=Hrt,  ylo=evlo,
                yhi=evhi,  xmultfactor=aut/1000,  xlabel="delay (fs)",
                ylabel="$\omega_{out} (eV)$", symmetricrange=True,
                cmap=tmpcmap, inpfontsize=inpfontsize,minzval=minzval,
                maxzval=maxzval, legend="no phase matching")#cmap="cubehelix")#cmap="rainbow")
if(savefigs):
    fign.savefig("transient_absorption_nophasematching_phase_0_0"+suffix+".png", bbox_inches="tight")

#make plots of transient absorption as a function of ns
nprval=1
for i in range(ns):
    #absorbarray[:,i,:]=2*imag(conjugate(phasevswarray[:,nprvec.index(0),i,:]) *phaseEwarray[:,0,0,:])
    absorbarray[:,i,:]=transientabsorption(kIRvswarray[:,nprvec.index(nprval),i,:], phaseEwarray[:,0,0,:])#2*imag(kIRvswarray[:,nprvec.index(1),i,:] *conjugate(phaseEwarray[:,0,0,:])/pow(abs(phaseEwarray[:,0,0,:]),2))

for i in range(ns):
    absorbarray[:,i,:]=maskarray(dtarray,warray,absorbarray[:,i,:],ylo=evlo,
                                 yhi=evhi, ymultfactor=Hrt)
maxval=abs(absorbarray).max()
print("maxval\t"+str(maxval))

for i in range(ns):
    fign=imshowplot(dtarray, warray, absorbarray[:, i, :],
                    ymultfactor=Hrt, legend="$n_{s}=$"+str(int(nsvec[i])),
                    ylo=evlo,  yhi=evhi,  xmultfactor=aut/1000,  xlabel="delay (fs)",
                    ylabel="$\omega_{out} (eV)$",symmetricrange=True,
                    absmax=maxval,cmap=tmpcmap, inpfontsize=inpfontsize,
                    maxzval=maxzval, minzval=minzval)
    if(savefigs):
        fign.savefig("transient_absorption_ns_" + str(int(nsvec[i])) +
                     "_wout_vs_dt" + suffix + ".png", bbox_inches="tight")
    #pcolorplot(dtarray,warray,absorbarray[:,i,:],inpcmapname='seismic')
    #plt.pcolormesh(array(dtvec),array(woutvec),transpose(absorbarray[:,i,:]),cmap='seismic')#

#
##transient absorption using quasienergies
#def quasienergy_contracted(w0):
#    tmparray=outer(nsvec,tmeasurevec)
#    tmparray=exp(-1j*w0*tmparray)
#    quasiarray=copy(kIRvstarray)
#    (n0,n1,n2,n3)=shape(kIRvstarray)
#    for i,j in iterproduct(range(n0),range(n1)):
#        for k,l in iterproduct(range(n2), range(n3)):
#            quasiarray[i,j,k,l]*=exp(-1j*w0*nsvec[k]*(tmeasurevec[l]-dtvec[i]))
#    quasift=nDFFT(quasiarray, axislist=[-1],shiftaxes=True)
#    quasifttrace=zeros((n0,n1,n3), dtype='complex')
#    for i in range(n2):
#        quasifttrace+=quasift[:,:,i,:]
#    return quasift, quasifttrace
#
##plot transient absorption using quasienergies
#quasift, quasifttrace=quasienergy_contracted(nm_to_eV(800)/Hrt)
#tmpabsorbarray=transientabsorption(quasifttrace[:,-1,:], phaseEwarray[:,0,0,:])
#for i in range(ns):
#    fign=imshowplot(dtarray, warray, quasift[:,-1, i, :],
#                    ymultfactor=Hrt, legend="quasienergy $n_{s}=$"+str(int(nsvec[i])),
#                    ylo=evlo,  yhi=evhi,  xmultfactor=aut/1000,  xlabel="delay (fs)",
#                    ylabel="$\omega_{out} (eV)$",symmetricrange=True,
#                    absmax=maxval,cmap=tmpcmap, inpfontsize=inpfontsize,
#                    maxzval=maxzval, minzval=minzval)
#fign=imshowplot(dtarray, warray, tmpabsorbarray, ymultfactor=Hrt,
#                legend="quasienergy transient absorption", ylo=evlo,  yhi=evhi,
#                xmultfactor=aut/1000,  xlabel="delay (fs)",
#                ylabel="$\omega_{out} (eV)$",symmetricrange=True,
#                absmax=maxval,cmap=tmpcmap, inpfontsize=inpfontsize,
#                maxzval=maxzval, minzval=minzval) 
plt.show()
