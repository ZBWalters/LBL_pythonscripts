from numpy import *
import re

#read phase matched dipole files, perform fourier transforms and write
#the results to files


def natural_sort(l): 
    #sort a list of strings such that numbers appear in numerical
    #order rather than the usual sorting convention
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def uniquevals(inplist):
    #return a list containing the set of unique elements of the input
    #list (not guaranteed to preserve order of the original list)
    return list(set(inplist))

def arrayfromfile(filename):
    #read a file generated by postprocess_phasematch.py and return an
    #array contining its information
    inparray=genfromtxt(filename,comments="#")
    #print( "shape inparray "+ str(shape(inparray)))
    (n,m)=shape(inparray)
    retarray=[]
    retarray.append(inparray[:,0])
    for i in range(1,int((m-1)/2)+1):
        retarray.append(inparray[:,2*i-1]+1j*inparray[:,2*i])
    return array(retarray)

def prependarraylist(arraylist):
    #prepend zeros to shorter dipole arrays so that the array list can
    #be joined together into a multidimensional array
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
    #call arrayfromfile in a particular directory
    fullfilename=str(fileindx)+"/"+filename
    return arrayfromfile(fullfilename)

def arraytofile(xarray,yarray,zarray,filename):
    #write an array to a file
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    for i in range(n):
        for j in range(m):
            outfile.write(str(xarray[i])+"\t"+str(yarray[j])+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

def doFFT(phasecol,fftcol,padarrays):
    #perform the relevant FFT
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



##############################################
#############Main program
#First, read in phasematch_parameterkey.txt 




#phasecol=6#column index for phase matching
fftcol=1
padarrays=False#zero pad arrays for purpose of interpolation?

(ncols,m)=shape(arrayfromfileindx(0,"ZDipoleexpect_phasematched.dat"))
for phasecol in range(1,ncols):
    doFFT(phasecol,fftcol,padarrays)