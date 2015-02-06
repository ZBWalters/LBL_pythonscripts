import glob
from numpy import *
import re
from scipy.linalg import solve
import subprocess
import shutil


def arrayfromfile(filename):
    inparray=genfromtxt(filename,comments="#")
    #print "shape inparray "+ str(shape(inparray))
    (n,m)=shape(inparray)
    retarray=[]
    retarray.append(inparray[:,0])
    retarray.append(inparray[:,1]+1j*inparray[:,2])
    retarray.append(inparray[:,3]+1j*inparray[:,4])
    retarray.append(inparray[:,5]+1j*inparray[:,6])
    return array(retarray)

def readarray(readindx):
    filename=str(readindx)+"/ZDipoleexpect.Dat"
    return arrayfromfile(filename)

def phasematcharrays(arraylist,colindx):
    #solve linear system for phase matched components
    (ncol,nt)=shape(arraylist[0])
    tarray=arraylist[0][0,:]
    nrhs=len(Rvecarray)
    retarray=zeros((nt,nrhs))*0j
    for i in range(nt):
        solnvec=solvelinsys(arraylist,colindx,i)
        for j in range(len(solnvec)):
            retarray[i,j]=solnvec[j]
    return tarray,retarray
    

def solvelinsys(arraylist,colindx,tindx):
    nrhs=len(arraylist)
    rhsvec=zeros(nrhs)*0j
    for i in range(nrhs):
        rhsvec[i]=arraylist[i][colindx,tindx]
    LHSmat=Linsys()
    #print("shape rhsvec\t",shape(rhsvec))
    #print("shape LHSmat\t",shape(LHSmat))
    solnvec=solve(LHSmat,rhsvec)
    return solnvec

#############

#CEParray=[[0,0,0],[0,0,.5],[.5,0,1],[.5,0,.5],[1,0,0],[1,0,.5],[1.5,0,1.5],[0,0,1.5],[0,0,1],[.5,0,1.5],[1.5,0,1],[1.5,0,.5]]
#these choices of CEPs are taken from 
#Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)
#CEParray=[[0,0,0],[0,0,.5],[.5,0,1]]
CEParray=[[0,0],[0,.5],[.5,0],[0,1]]
#first three from Meyer & Engel paper

Rvecarray=[[1,0],[0,1],[1,1],[1,-1]]
#Rvecarray=[[1,-1,0],[1,0,-1],[0,1,-1]]
#Rvecarray=[[1,0,0],[0,1,0],[0,0,1],[1,-1,1],[1,1,-1],[-1,1,1],[2,-1,0],[2,0,-1],[-1,2,0],[0,2,-1],[-1,0,2],[0,-1,2]]
#Rvecarray taken from Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)
#Rvecarray[i].[k1,k2,k3]=k vector for phase matched component

def Linsys():
    Lmat=zeros((len(CEParray),len(Rvecarray)))*0j
    for i in range(len(CEParray)):
        for j in range(len(CEParray)):
            Lmat[i,j]=exp(1j*pi*dot(CEParray[i],Rvecarray[j]))
    return Lmat

###################################
def writephasematchedvalues(phasematchdirstr,indx,tarray,phasematcharray):
    dirstr=phasematchdirstr+str(indx)+"/"
    subprocess.call(["mkdir",dirstr])
    filename=dirstr+"ZDipoleexpect_phasematched.dat"
    f=open(filename,'w')
    f.write("#t\tk1\tk2\tk1+k2\tk1-k2\n")
    for i in range(len(tarray)):
        tmpstr=str(real(tarray[i]))
        for j in range(len(phasematcharray[i,:])):
            tmpstr+="\t"+str(real(phasematcharray[i,j]))+"\t"+str(imag(phasematcharray[i,j]))
        tmpstr+="\n"
        f.write(tmpstr)
    f.close()

def writephasematchkey(filename,indx,paramarray):
    t1=paramarray[indx*nCEP,1]
    t2=paramarray[indx*nCEP,2]
    filename.write(str(indx)+"\t"+str(t1)+"\t"+str(t2)+"\n")


####################################
####################################
####################################

paramarray=genfromtxt("parameterkey.txt")
(nparam,mparam)=shape(paramarray)
nCEP=len(CEParray)
ntvals=int(nparam/nCEP)#number of values of t1,t2 for which we
                           #have to find phase matched components

phasematchdirstr="phasematched/"
subprocess.call(["mkdir",phasematchdirstr])

keyfilestr=phasematchdirstr+"phasematch_parameterkey.txt"
keyfile=open(keyfilestr,'w')
keyfile.write("#indx\tdeltat1\tdeltat2\n")

for i in range(ntvals):
    arraylist=[]
    for j in range(nCEP):
        readindx=i*nCEP+j
        arraylist.append(readarray(readindx))
    tarray,phasematcharray=phasematcharrays(arraylist,2)
    writephasematchedvalues(phasematchdirstr,i,tarray,phasematcharray)
    writephasematchkey(keyfile,i,paramarray)


keyfile.close()
