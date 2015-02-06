import numpy,scipy,pylab
from numpy import *
from scipy import *
from pylab import *

def plotchrom(data,title):
    names=['t','free',r'$\alpha_{19A}$',r'$\alpha_{19B}$',r'$\beta_{50C}$',r'$\beta_{82C}$',r'$\beta_{158C}$',r'$\beta_{50D}$',r'$\beta_{82D}$',r'$\beta_{158D}$']
    #styles=['','','b--','b:','k--','r--','r:','k:','r-.','g--']
    styles=['','','r--','r:','b--','g--','g:','b:','g-.','y--']
    (nt,nc)=shape(data)
    figure()
    #suptitle(title)
    for i in range(2,nc):
        plot(data[:,0],data[:,i],styles[i],label=names[i])
    legend(loc='lower right')

def plotlines(data,title):
    (nt,nc)=shape(data)
    figure()
    suptitle(title)
    for i in range(1,nc):
        plot(data[:,0],data[:,i],label=str(i-1))
    legend(loc='best')

def logplotlines(data,title):
    (nt,nc)=shape(data)
    figure()
    semilogy()
    suptitle(title)
    for i in range(1,nc):
        plot(data[:,0],data[:,i],label=str(i))
    legend(loc='best')

def logplotchrom(data,title):
    names=['t','free',r'$\alpha_{19A}$',r'$\alpha_{19B}$',r'$\beta_{50C}$',r'$\beta_{82C}$',r'$\beta_{158C}$',r'$\beta_{50D}$',r'$\beta_{82D}$',r'$\beta_{158D}$']
    #styles=['','','b--','b:','k--','r--','r:','k:','r-.','g--']
    styles=['','','r--','r:','b--','g--','g:','b:','g-.','y--']
    (nt,nc)=shape(data)
    figure()
    #suptitle(title)
    semilogy()
    for i in range(2,nc):
        plot(data[:,0],data[:,i],styles[i],label=names[i])
    legend(loc='lower right')
        
def plotchromfile(filename):
    data=array(loadtxt(filename))
    plotchrom(data,filename)
    show()

def logplotchromfile(filename):
    data=array(loadtxt(filename))
    logplotchrom(data,filename)
    show()

def plotfile(filename):
    #figure()
    #suptitle(filename)
    data=array(loadtxt(filename))
    #plot(data)
    plotlines(data,filename)
    show()
    #savefig(filename+'.png')

def plotfiles(filelist):
    for file in filelist:
        plotfile(file)
    return

def logplotfile(filename):
    #figure()
    #suptitle(filename)
    #semilogy()
    data=array(loadtxt(filename))
    logplotlines(data,filename)
    #plot(data)
    show()

def logplotfiles(filelist):
    for file in filelist:
        logplotfile(file)
    return

#def weightchromdat(data,temp):
#    newdata=data*0.
#    Hcm=219470.0
#    #Estate=[3.15,3.04,3.47,3.25,3.30,3.53,3.28,3.38]
#    #Estate=[.11, 0., .43, .21, .26, .49, .24, .34]
#    Estate=[12410,12530,12210,12320,12480,12630,12440,12700]
#    Estate=list(array(Estate)/Hcm)
#    kb=8.617343e-5
#    kbT=kb*temp
#    print "shape(data)",shape(data), shape(data)[0],shape(data)[1]
#    for i in range(shape(data)[0]):
#        Z=0.
#        Tr=0.
#        newdata[i,0]=data[i,0]
#        newdata[i,1]=data[i,1]
#        for j in range(8):
#            Z=Z+data[i,j+2]*exp(-Estate[j]/kbT)
#            Tr=Tr+data[i,j+2]
#        for j in range(8):
#            newdata[i,j+2]=data[i,j+2]*exp(-Estate[j]/kbT)*Tr/Z
#    return newdata
#
#def weightplotchromfile(filename,temp):
#    data=array(loadtxt(filename))
#    newdata=weightchromdat(data,temp)
#    plotchrom(newdata,filename)
#    show()
#
#def weightlogplotchromfile(filename,temp):
#    data=array(loadtxt(filename))
#    newdata=weightchromdat(data,temp)
#    logplotchrom(newdata,filename)
#    show()

def reweight_pops(kbT,trialE,pops):
    totpop=0.
    minE=min(trialE)
    Etmp=trialE-minE
    retpops=pops*0.
    for i in range(len(trialE)):
        retpops[i]=pops[i]*exp(-Etmp[i]/(kbT))
    totpop=sum(retpops)
    #print "totpop",totpop
    retpops=retpops/totpop
    return retpops

def reweight_amps(kbT,trialE,amps):
    totpop=0.
    minE=min(trialE)
    Etmp=trialE-minE
    retamps=amps*0.
    totpop=0.
    for i in range(len(trialE)):
        retamps[i]=amps[i]*exp(-Etmp[i]/(2*kbT))
        totpop=totpop+retamps[i]**2.
    #print "totpop",totpop
    retamps=retamps/sqrt(totpop)
    return retamps

def weightdat(kbT,trialE,dat):
    retdat=dat*0.
    (nrows,ncols)=shape(dat)
    for i in range(nrows):
        retdat[i,0]=dat[i,0]
        retdat[i,1:]=reweight_pops(kbT,trialE,dat[i,1:])
    return retdat

def weightplotfile(filename,trialE,kbT):
    data=array(loadtxt(filename))
    newdata=weightdat(kbT,trialE,data)
    #print "newdata",shape(newdata)
    #plotchrom(newdata,filename)
    plotlines(newdata,filename+"_reweighted")
    show()
    #savefig(filename+"_reweighted"+'.png')

def weightplotfiles(filelist,trialE,kbT):
    for file in filelist:
        weightplotfile(file,trialE,kbT)
    return

def weightlogplotfile(filename,trialE,kbT):
    data=array(loadtxt(filename))
    newdata=weightdat(kbT,trialE,data)
    #logplotchrom(newdata,filename)
    logplotlines(newdata,filename+"_reweighted")
    show()
    #savefig(filename+"_reweighted_log"+'.png')
