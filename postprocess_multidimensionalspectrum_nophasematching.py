from numpy import *
import re

#######################
#This script takes the fourier transform of the calculated dipoles
#before phase matching is applied.
#
#######################


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def uniquevals(inplist):
    return list(set(inplist))

#def arrayfromfile(filename):
#    inparray=genfromtxt(filename,comments="#")
#    #print( "shape inparray "+ str(shape(inparray)))
#    (n,m)=shape(inparray)
#    retarray=[]
#    retarray.append(inparray[:,0])
#    for i in range(1,int((m-1)/2)+1):
#        retarray.append(inparray[:,2*i-1]+1j*inparray[:,2*i])
#    return array(retarray)

def arrayfromfile(filename,rvals=False):
    inparray=genfromtxt(filename,comments="#")
    #print "shape inparray "+ str(shape(inparray))
    (n,m)=shape(inparray)
    retarray=[]
    if(rvals):#return only the real components
        retarray.append(inparray[:,0]+0j)
        retarray.append(inparray[:,1]+0j)
        retarray.append(inparray[:,3]+0j)
        retarray.append(inparray[:,5]+0j)       
    else:#return the real and imaginary components
        retarray.append(inparray[:,0])
        retarray.append(inparray[:,1]+1j*inparray[:,2])
        retarray.append(inparray[:,3]+1j*inparray[:,4])
        retarray.append(inparray[:,5]+1j*inparray[:,6])
    return array(retarray)


def prependarraylist(arraylist):
    retlist=[]
    #find length of longest calculation
    maxn=0
    for i in range(len(arraylist)):
        (m,n)=shape(arraylist[i])
        maxn=max(maxn,n)
    for i in range(len(arraylist)):
        tmpvec=arraylist[i]
        #print("shape tmpvec\t"+str(shape(tmpvec))+"\t"+str(maxn))
        (m,n)=shape(tmpvec)
        newvec=zeros((m,maxn))*0j
        for a in range(m):
            newvec[a,(maxn-n):]=array(tmpvec)[a,:]
        retlist.append(newvec)
    return retlist
    

def arrayfromfileindx(fileindx,filename):
    fullfilename=str(fileindx)+"/"+filename
    return arrayfromfile(fullfilename)

def arraytofile(xarray,yarray,zarray,filename):
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    for i in range(n):
        for j in range(m):
            outfile.write(str(real(xarray[i]))+"\t"+str(real(yarray[j]))+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

def doFFT(phasecol,fftcol,padarrays):
    parameterarray=genfromtxt("phasematch_parameterkey.txt")
    nt1=len(uniquevals(parameterarray[:,1]))#number of t1 values (FT over this)
    nt2=len(uniquevals(parameterarray[:,2]))#number of t2 values
    t1array=sort(uniquevals(parameterarray[:,1]))
    t2array=sort(uniquevals(parameterarray[:,2]))

    arraylist=[]
    for i in range(nt1):
        tmplist=[]
        for j in range(nt2):
            indx=j*nt1+i
            tmparray=arrayfromfileindx(indx,"ZDipoleexpect_phasematched.dat")
            t3array=real(tmparray[0,:])
            nt3=len(t3array)
            tmpvec=tmparray[phasecol,:]
            tmplist.append(tmpvec)
        arraylist.append(tmplist)

    arraylist=prependarraylist(arraylist)#prepend initial zeros so that arrays
                               #have same length
    inpdiparray=array(arraylist)
    
    #if padarrays==True, then zero pad the fourier transforms 
    npad=512#256
    if(padarrays):
        np1=npad
        np2=npad
        np3=npad
    else:
        np1=nt1
        np2=nt2
        np3=nt3
    
    
    ####Perform 2D Fourier Transforms
    
    if(fftcol==1):#fourier transform t1 column
        arraytofile(t1array,t3array,inpdiparray[:,0,:],"twoddiparray."+str(phasecol)+".dat")
    
        twodfreqarray=fft.fft2(inpdiparray,(np1,np3),(0,2))
        twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,2))
        
    
        dt1=t1array[1]-t1array[0]
        dt3=t3array[1]-t3array[0]
        fftfreq1=fft.fftfreq(np1,dt1)*2*pi
        fftfreq3=fft.fftfreq(np3,dt3)*2*pi
        fftfreq1.sort()
        fftfreq3.sort()
        dw1=fftfreq1[1]-fftfreq1[0]
        dw3=fftfreq3[1]-fftfreq3[0]
    
        arraytofile(fftfreq1,fftfreq3,twodfreqarray[:,0,:],"twodfreqarray."+str(phasecol)+".dat")
    
    
    if(fftcol==2):#fourier transform t2 column
        arraytofile(t2array,t3array,inpdiparray[0,:,:],"twoddiparray."+str(phasecol)+".dat")
    
        twodfreqarray=fft.fft2(inpdiparray,(np2,np3),(1,2))
        twodfreqarray=fft.fftshift(twodfreqarray,axes=(1,2))
        
    
    
        dt2=t2array[1]-t2array[0]
        dt3=t3array[1]-t3array[0]
        fftfreq2=fft.fftfreq(np2,dt2)*2*pi
        fftfreq3=fft.fftfreq(np3,dt3)*2*pi
        fftfreq2.sort()
        fftfreq3.sort()
        dw2=fftfreq2[1]-fftfreq2[0]
        dw3=fftfreq3[1]-fftfreq3[0]
        
        arraytofile(fftfreq2,fftfreq3,twodfreqarray[0,:,:],"twodfreqarray."+str(phasecol)+".dat")
##########################
def readarray(readindx,rvals=False):
    filename=str(readindx)+"/ZDipoleexpect.Dat"
    return arrayfromfile(filename,rvals)

#def readdipoles(nt1vals,cepindx):
#    arraylist=[]
#    for i in range(nt1vals):
#        readindx=i*nCEP+cepindx
#        arraylist.append(readarray(readindx,True))
#    t3array=arraylist[0],[0,:]
#    nt3vals=len(tarray)
#    retarray=zeros((nt1vals,nt3vals))
#    for i in range(nt1vals):
#        for j in range(nt3vals):
#            retarray[i,j]=arraylist[i][2,j]
#    return t3array,retarray

def readbigdipolearray(nparam,colindx):
    arraylist=[]
    t3array=readarray(0,True)[0,:]
    for i in range(nparam):
        arraylist.append(readarray(i,True)[colindx,:])
    retarray=array(arraylist)
    return t3array,retarray
    
    
#############################################
CEParray=[[0,0,0],[0,0,.5],[.5,0,1],[.5,0,.5],[1,0,0],[1,0,.5],[1.5,0,1.5],[0,0,1.5],[0,0,1],[.5,0,1.5],[1.5,0,1],[1.5,0,.5]]
#these choices of CEPs are taken from 
#Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)

Rvecarray=[[1,0,0],[0,1,0],[0,0,1],[1,-1,1],[1,1,-1],[-1,1,1],[2,-1,0],[2,0,-1],[-1,2,0],[0,2,-1],[-1,0,2],[0,-1,2]]
#Rvecarray taken from Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)
#Rvecarray[i].[k1,k2,k3]=k vector for phase matched component

def Linsys():
    Lmat=zeros((len(CEParray),len(Rvecarray)))*0j
    for i in range(len(CEParray)):
        for j in range(len(CEParray)):
            Lmat[i,j]=exp(1j*pi*dot(CEParray[i],Rvecarray[j]))
    return Lmat



##############################################
padarrays=False#do not zero pad for purposes of interpolation

paramarray=genfromtxt("parameterkey.txt")
(nparam,mparam)=shape(paramarray)
nCEP=len(CEParray)
ntvals=int(nparam/nCEP)#number of values of t1,t2 for which we
                           #have to find phase matched components
t1array=sort(uniquevals(paramarray[:,1]))
nt1=len(t1array)
#print("t1array\t"+str(t1array))

t3array,bigdipolearray=readbigdipolearray(nparam,2)
nt3=len(t3array)

for cepindx in range(len(CEParray)):
    diparray=zeros((nt1,nt3))*0j
    for i in range(nt1):
        indx=i*nCEP+cepindx
        diparray[i,:]=bigdipolearray[indx,:]
    #write dipoles to file
    arraytofile(t1array,t3array,diparray,"twoddiparray_nonphasematched."+str(cepindx)+".dat")
    
    #if padarrays==True, then zero pad the fourier transforms 
    npad=512#256
    if(padarrays):
        np1=npad
        np3=npad
    else:
        np1=nt1
        np3=nt3


    #do fft

    twodfreqarray=fft.fft2(diparray,(np1,np3),(0,1))
    twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,1))
    dt1=t1array[1]-t1array[0]
    dt3=t3array[1]-t3array[0]
    fftfreq1=fft.fftfreq(np1,dt1)*2*pi
    fftfreq3=fft.fftfreq(np3,dt3)*2*pi
    fftfreq1.sort()
    fftfreq3.sort()
    dw1=fftfreq1[1]-fftfreq1[0]
    dw3=fftfreq3[1]-fftfreq3[0]
    
    arraytofile(fftfreq1,fftfreq3,twodfreqarray,"twodfreqarray_nonphasematched."+str(cepindx)+".dat")

##phasecol=6#column index for phase matching
#fftcol=1
#padarrays=False#zero pad arrays for purpose of interpolation?
#
#(ncols,m)=shape(arrayfromfileindx(0,"ZDipoleexpect_phasematched.dat"))
#for phasecol in range(1,ncols):
#    doFFT(phasecol,fftcol,padarrays)
