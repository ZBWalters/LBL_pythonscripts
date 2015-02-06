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
            outfile.write(str(xarray[i])+"\t"+str(yarray[j])+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

def doIonizationFFT(phasecol,padarrays=False):
    parameterarray=genfromtxt("phasematch_parameterkey.txt")
    nt1=len(uniquevals(parameterarray[:,1]))
    t1array=sort(uniquevals(parameterarray[:,1]))

    arraylist=[]
    for i in range(nt1):
        indx=i
        tmparray=arrayfromfileindx(indx,"ZDipoleexpect_phasematched.dat")
        t2array=real(tmparray[0,:])
        nt2=len(t2array)
        tmpvec=tmparray[phasecol,:]
        arraylist.append(tmpvec)
    inpdiparray=array(arraylist)

    #if padarrays==True, then zero pad the fourier transforms 
    npad=512#256
    if(padarrays):
        np1=npad
        np2=npad
    else:
        np1=nt1
        np2=nt2


    #do fourier transforms
    arraytofile(t1array,t2array,inpdiparray,"twoddiparray."+str(phasecol)+".dat")
    
    twodfreqarray=fft.fft2(inpdiparray,(np1,np2),(0,1))
    twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,1))
    
    
    dt1=t1array[1]-t1array[0]
    dt2=t2array[1]-t2array[0]
    fftfreq1=fft.fftfreq(np1,dt1)*2*pi
    fftfreq2=fft.fftfreq(np2,dt2)*2*pi
    fftfreq1.sort()
    fftfreq2.sort()
    dw1=fftfreq1[1]-fftfreq1[0]
    dw2=fftfreq2[1]-fftfreq2[0]
    
    arraytofile(fftfreq1,fftfreq2,twodfreqarray,"twodfreqarray."+str(phasecol)+".dat")

    #next, do 1D fourier transforms
    t1freq2array=fft.fft(inpdiparray,np1,0)
    t1freq2array=fft.fftshift(t1freq2array,0)

    arraytofile(t1array,fftfreq2,t1freq2array,\
                "t1freq2array."+str(phasecol)+".dat")

    freq1t2array=fft.fft(inpdiparray,np2,1)
    freq1t2array=fft.fftshift(freq1t2array,1)

    arraytofile(t1array,fftfreq2,freq1t2array,\
                "freq1t2array."+str(phasecol)+".dat")


#def doFFT(phasecol,fftcol,padarrays):
#    parameterarray=genfromtxt("phasematch_parameterkey.txt")
#    nt1=len(uniquevals(parameterarray[:,1]))#number of t1 values (FT over this)
#    nt2=len(uniquevals(parameterarray[:,2]))#number of t2 values
#    t1array=sort(uniquevals(parameterarray[:,1]))
#    t2array=sort(uniquevals(parameterarray[:,2]))
#
#    arraylist=[]
#    for i in range(nt1):
#        tmplist=[]
#        for j in range(nt2):
#            indx=j*nt1+i
#            tmparray=arrayfromfileindx(indx,"ZDipoleexpect_phasematched.dat")
#            t3array=real(tmparray[0,:])
#            nt3=len(t3array)
#            tmpvec=tmparray[phasecol,:]
#            tmplist.append(tmpvec)
#        arraylist.append(tmplist)
#
#    arraylist=prependarraylist(arraylist)#prepend initial zeros so that arrays
#                               #have same length
#    inpdiparray=array(arraylist)
#    
#    #if padarrays==True, then zero pad the fourier transforms 
#    npad=512#256
#    if(padarrays):
#        np1=npad
#        np2=npad
#        np3=npad
#    else:
#        np1=nt1
#        np2=nt2
#        np3=nt3
#    
#    
#    ####Perform 2D Fourier Transforms
#    
#    if(fftcol==1):#fourier transform t1 column
#        arraytofile(t1array,t3array,inpdiparray[:,0,:],"twoddiparray."+str(phasecol)+".dat")
#    
#        twodfreqarray=fft.fft2(inpdiparray,(np1,np3),(0,2))
#        twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,2))
#        
#    
#        dt1=t1array[1]-t1array[0]
#        dt3=t3array[1]-t3array[0]
#        fftfreq1=fft.fftfreq(np1,dt1)*2*pi
#        fftfreq3=fft.fftfreq(np3,dt3)*2*pi
#        fftfreq1.sort()
#        fftfreq3.sort()
#        dw1=fftfreq1[1]-fftfreq1[0]
#        dw3=fftfreq3[1]-fftfreq3[0]
#    
#        arraytofile(fftfreq1,fftfreq3,twodfreqarray[:,0,:],"twodfreqarray."+str(phasecol)+".dat")
#    
#    
#    if(fftcol==2):#fourier transform t2 column
#        arraytofile(t2array,t3array,inpdiparray[0,:,:],"twoddiparray."+str(phasecol)+".dat")
#    
#        twodfreqarray=fft.fft2(inpdiparray,(np2,np3),(1,2))
#        twodfreqarray=fft.fftshift(twodfreqarray,axes=(1,2))
#        
#    
#    
#        dt2=t2array[1]-t2array[0]
#        dt3=t3array[1]-t3array[0]
#        fftfreq2=fft.fftfreq(np2,dt2)*2*pi
#        fftfreq3=fft.fftfreq(np3,dt3)*2*pi
#        fftfreq2.sort()
#        fftfreq3.sort()
#        dw2=fftfreq2[1]-fftfreq2[0]
#        dw3=fftfreq3[1]-fftfreq3[0]
#        
#        arraytofile(fftfreq2,fftfreq3,twodfreqarray[0,:,:],"twodfreqarray."+str(phasecol)+".dat")    
#


##############################################
#First, read in phasematch_parameterkey.txt 




#phasecol=6#column index for phase matching
fftcol=1
padarrays=False#zero pad arrays for purpose of interpolation?

(ncols,m)=shape(arrayfromfileindx(0,"ZDipoleexpect_phasematched.dat"))
for phasecol in range(1,ncols):
    #doFFT(phasecol,fftcol,padarrays)
    doIonizationFFT(phasecol,padarrays)
