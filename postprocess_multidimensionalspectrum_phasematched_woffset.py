from numpy import *
import re



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def uniquevals(inplist):
    return list(set(inplist))

def arrayfromfile(filename):
    inparray=genfromtxt(filename,comments="#")
    #print( "shape inparray "+ str(shape(inparray)))
    (n,m)=shape(inparray)
    retarray=[]
    retarray.append(inparray[:,0])
    for i in range(1,int((m-1)/2)+1):
        retarray.append(inparray[:,2*i-1]+1j*inparray[:,2*i])
    return array(retarray)

def arrayfromfileindx(fileindx,filename):
    fullfilename=str(fileindx)+"/"+filename
    return arrayfromfile(fullfilename)

def arraytofile(xarray,yarray,zarray,filename):
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    for i in range(n):
        for j in range(m):
            outfile.write(str(xarray[i])+"\t"+str(yarray[j])+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

def recenterdips(t1array,t2array,diparray,wlist):
    retarray=diparray*1.
    for i in range(len(t1array)):
        for j in range(len(t2array)):
            retarray[i,j]*=exp(-1j*(t1array[i]*wlist[0]+t2array[j]*wlist[1]))
    return retarray

def doFFT(phasecol,fftcol,padarrays,wlist=[0,0]):
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
        tmpdiparray=recenterdips(t1array,t3array,inpdiparray[:,0,:],wlist)
        arraytofile(t1array,t3array,tmpdiparray,"twoddiparray."+str(phasecol)+".dat")
    
        twodfreqarray=fft.fft2(tmppdiparray,(np1,np3),(0,1))
        twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,1))
        
    
        dt1=t1array[1]-t1array[0]
        dt3=t3array[1]-t3array[0]
        fftfreq1=fft.fftfreq(np1,dt1)*2*pi
        fftfreq3=fft.fftfreq(np3,dt3)*2*pi
        fftfreq1.sort()
        fftfreq3.sort()
        dw1=fftfreq1[1]-fftfreq1[0]
        dw3=fftfreq3[1]-fftfreq3[0]
    
        arraytofile(fftfreq1,fftfreq3,twodfreqarray,"twodfreqarray."+str(phasecol)+".dat")
    
    
    if(fftcol==2):#fourier transform t2 column
        tmpdiparray=recenterdips(t2array,t3array,inpdiparray[0,:,:],wlist)
        arraytofile(t2array,t3array,tmpdiparray,"twoddiparray."+str(phasecol)+".dat")
    
        twodfreqarray=fft.fft2(tmpdiparray,(np2,np3),(0,1))
        twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,1))
        
    
    
        dt2=t2array[1]-t2array[0]
        dt3=t3array[1]-t3array[0]
        fftfreq2=fft.fftfreq(np2,dt2)*2*pi
        fftfreq3=fft.fftfreq(np3,dt3)*2*pi
        fftfreq2.sort()
        fftfreq3.sort()
        dw2=fftfreq2[1]-fftfreq2[0]
        dw3=fftfreq3[1]-fftfreq3[0]
        
        arraytofile(fftfreq2,fftfreq3,twodfreqarray,"twodfreqarray."+str(phasecol)+".dat")    



##############################################
#First, read in phasematch_parameterkey.txt 




#phasecol=6#column index for phase matching
fftcol=2
padarrays=False#zero pad arrays for purpose of interpolation?

(ncols,m)=shape(arrayfromfileindx(0,"ZDipoleexpect_phasematched.dat"))
Hrt=27.21
w1=115/Hrt
w2=(5.3+10.6)/(2*Hrt)
warray=[[0,w1],[0,0],[0,0],[0,w1],[0,w1],[0,w1],[0,w1],[0,w1],[0,w1],[0,0],[0,w1],[0,0]]
for phasecol in range(1,ncols):
    doFFT(phasecol,fftcol,padarrays,warray[phasecol-1])
