import glob
from numpy import *
#import scipy
#from natsort import natsorted
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def uniquevals(inplist):
    return list(set(inplist))

def arrayfromfile(filename):
    inparray=genfromtxt(filename,comments="#")
    #print "shape inparray "+ str(shape(inparray))
    (n,m)=shape(inparray)
    retarray=[]
    retarray.append(inparray[:,0])
    for i in range(1,int((m-1)/2)):
        retarray.append(inparray[:,2*i-1]+1j*inparray[:,2*i])
    return array(retarray)

def arrayfromfileindx(fileindx,filename):
    fullfilename=str(fileindx)+"/"+filename
    return arrayfromfile(fullfilename)

def mergearrays(arraylist,colindx):
    tarray=real(arraylist[0][0,:])
    retlist=[]
    for i in range(len(arraylist)):
        arry=arraylist[i]
        retlist.append(arry[colindx,:])
    return tarray, array(retlist)

def listtostring(lst):
    retstr=str(real(lst[0]))
    n=len(lst)
    for i in range(n)[1:]:
        retstr=retstr+"\t"+str(real(lst[i]))+"\t"+str(imag(lst[i]))
    retstr=retstr+"\n"
    return retstr

def arraytofile(xarray,yarray,zarray,filename):
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    for i in range(n):
        for j in range(m):
            outfile.write(str(xarray[i])+"\t"+str(yarray[j])+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

def zeropadarrays(xarray,yarray,zarray,newn,newm):#(newn,newm)):
    n=len(xarray)
    m=len(yarray)

    nmid=n/2
    nmidnew=newn/2
    mmid=m/2
    mmidnew=newm/2

    dx=xarray[1]-xarray[0]
    dy=yarray[1]-yarray[0]

    #now want to construct x array with same dx & same center
    retx=zeros(newn)
    rety=zeros(newm)
    midxval=xarray[nmid]
    midyval=yarray[mmid]
    for i in range(newn):
        retx[i]=midxval+(i-nmidnew)*dx
    for j in range(newm):
        rety[j]=midyval+(j-mmidnew)*dy
    
    retz=zeros((newn,newm))*0j
    for i in range(n):
        idiff=nmid-i
        for j in range(m):
            jdiff=mmid-j
            retz[nmidnew-idiff,mmidnew-jdiff]=zarray[i,j]
    
    return [retx,rety,retz]

def lowpassfilter(freq1,freq2,array,xlim,ylim):#(xlim,ylim)):
    (n,m)=shape(array)
    retarray=array*1.
    for i in range(n):
        for j in range(m):
            if(abs(freq1[i])>xlim or abs(freq2[j])>ylim):
                retarray[i,j]=0.
    return retarray
    

##############################################
#First, read in phasematch_parameterkey.txt 

parameterarray=genfromtxt("phasematch_parameterkey.txt")
nt1=len(uniquevals(parameterarray[:,1]))#number of t1 values (FT over this)
nt2=len(uniquevals(parameterarray[:,2]))#number of t2 values

dt1=parameterarray[1,1]-parameterarray[0,1]#time step to use in FT

#For every value of t2, take the fourier transform of the desired
#polarization vector
polcol=4#column index of polarization column
for j in range(nt2):
    arraylist=[]
    for i in range(nt1):
        indx=j*nt1+i
        arraylist.append(arrayfromfileindx(indx,"ZDipoleexpect_phasematched.dat"))
    inpt2array,inptwodtarray=mergearrays(arraylist,polcol)
    inptwodtarray=2*real(inptwodtarray)
    dt2=inpt2array[1]-inpt2array[0]
    inpt1array=range(nt1)*dt1

    [t1array,t2array,twodtarray]=zeropadarrays(inpt1array,inpt2array,inptwodtarray,256,256)
    
    
    fftfreq1=fft.fftfreq(len(t1array),dt1)*2*pi
    fftfreq2=fft.fftfreq(len(t2array),dt2)*2*pi
    fftfreq1.sort()
    fftfreq2.sort()
    dw1=fftfreq1[1]-fftfreq1[0]
    dw2=fftfreq2[1]-fftfreq2[0]
    
    
    arraytofile(inpt1array,inpt2array,inptwodtarray,"twoddiparray.dat")
    
    
    
    
    
    twodfreqarray=fft.fft2(twodtarray)
    twodfreqarray=fft.fftshift(twodfreqarray)
    (n,m)=shape(twodfreqarray)
    for i in range(n):
        for j in range(m):
            twodfreqarray[i,j]=twodfreqarray[i,j]*pow(-1,i-j)
    
    #print twodfreqarray
    
    
    arraytofile(fftfreq1,fftfreq2,twodfreqarray,"twodfreqarray.dat")
    
    #now *reverse* the fourier transforms of each dimension to get time resolved
    #pictures
    tauarray1=fft.fftfreq(len(fftfreq1),dw1)*(2*pi)
    tauarray2=fft.fftfreq(len(fftfreq2),dw2)*(2*pi)
    tauarray1.sort()
    tauarray2.sort()
    dtau1=tauarray1[1]-tauarray1[0]
    dtau2=tauarray2[1]-tauarray2[0]
    
    #reverse ft for first dimension
    n=len(fftfreq1)
    m=len(fftfreq2)
    tau1f2array=zeros((n,m))*0j
    for j in range(m):
        tmparray=twodfreqarray[:,j]
        tmparray=fft.fft(tmparray)
        tmparray=fft.fftshift(tmparray)
        for i in range(len(tmparray)):
            tmparray[i]*=pow(-1,i)
        tau1f2array[:,j]=tmparray
    
    arraytofile(tauarray1,fftfreq2,tau1f2array,"tau1f2array.dat")
    
    #reverse ft for second dimension
    n=len(fftfreq1)
    m=len(fftfreq2)
    f1tau2array=zeros((n,m))*0j
    for i in range(n):
        tmparray=twodfreqarray[i,:]
        tmparray=fft.fft(tmparray)
        tmparray=fft.fftshift(tmparray)
        for j in range(len(tmparray)):
            tmparray[j]*=pow(-1,j+1)
        f1tau2array[i,:]=tmparray
    
    arraytofile(fftfreq1,tauarray2,f1tau2array,"f1tau2array.dat")
    



