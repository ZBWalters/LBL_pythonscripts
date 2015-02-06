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

def arraytofile(xarray,yarray,zarray,filename):
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    for i in range(n):
        for j in range(m):
            outfile.write(str(xarray[i])+"\t"+str(yarray[j])+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
            
            #outfile.write(str(i)+"\t"+str(j)+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

def FFTfreqarray(dt,npts):
    #nyquist critical frequency is 2 points per cycle
    #so that wcrit*(2*dt)=2*pi
    wcrit=2*pi/(2*dt)
    dw=wcrit/npts
    return array(range(npts))*dw-wcrit/2

def twoDFFT(t1array,t2array,twoDdiparray):
    (np1,np2)=shape(twoDdiparray)
    print("twoDFFT shape twoDdiparray "+str(shape(twoDdiparray)))
    twodfreqarray=fft.fft2(twoDdiparray,(np1,np2),(0,1))
    twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,1))
    print("twoDFFT shape twodfreqarray"+str(shape(twodfreqarray)))

#    #powers of -1 arising from shift
#    (n,l,m)=shape(twodfreqarray)
#    for i in range(n):
#        for j in range(m):
#            for k in range(l):
#                twodfreqarray[i,k,j]*=1.#pow(-1.,j)

    dt1=t1array[1]-t1array[0]
    dt2=t2array[1]-t2array[0]
    #fftfreq1=fft.fftfreq(np1,dt1)*2*pi
    #fftfreq2=fft.fftfreq(np2,dt2)*2*pi
    #fftfreq1=sort(fftfreq1)
    #fftfreq2=sort(fftfreq2)
    fftfreq1=FFTfreqarray(dt1,np1)
    fftfreq2=FFTfreqarray(dt2,np2)
    dw1=fftfreq1[1]-fftfreq1[0]
    dw2=fftfreq2[1]-fftfreq2[0]
    #print ("fftfreq1\t"+str(fftfreq1))
    #print ("fftfreq2\t"+str(fftfreq2))
    return [fftfreq1,fftfreq2,twodfreqarray]

def padFFT(t1array,t2array,twoDdiparray,npad1,npad2):
    n1,n2=shape(twoDdiparray)
    print("n1, n2 "+str(n1)+"\t"+str(n2))
    print("npad1, npad2 "+str(npad1)+"\t"+str(npad2))

    dt1=t1array[1]-t1array[0]
    dt2=t2array[1]-t2array[0]
    fftfreq1=fft.fftfreq(n1,dt1)#*2*pi
    fftfreq2=fft.fftfreq(n2,dt2)#*2*pi
    dw1=fftfreq1[1]-fftfreq1[0]
    dw2=fftfreq2[1]-fftfreq2[0]


    retfftarray=zeros((n1*npad1,n2*npad2))*0j
    print("shape of retfftarray "+str(shape(retfftarray)))

    for i in range(npad1):
        for j in range(npad2):
            phase1=exp(-1j*(dw1/npad1)*dt1*i)#exp(1j*.5*pi/(npad1*n1)*i)#exp(-1j*pi*i/(npad1))
            phase2=exp(-1j*(dw2/npad2)*dt2*j)#exp(1j*.5*pi/(npad2*n2)*j)#exp(-1j*pi*j/(npad2))
            tmpdiparray=zeros((n1,n2))*0j
            for k in range(n1):
                for l in range(n2):
                    tmpdiparray[k,l]=twoDdiparray[k,l]*\
                                      pow(phase1,k)*pow(phase2,l)
            [fftfreq1,fftfreq2,tmptwodfreqarray]=twoDFFT(t1array,t2array,\
                                                         tmpdiparray)
            arraytofile(fftfreq1,fftfreq2,tmptwodfreqarray,"tmp."+str(i)+"."+str(j)+".dat")
            for k in range(n1):
                for l in range(n2):
                    indx1=k*npad1+i
                    indx2=l*npad2+j
                    retfftarray[indx1,indx2]=tmptwodfreqarray[k,l]

#    for i in range(npad1):
#        for j in range(npad2):
#            tmpdiparray=zeros((n1,n2))*0j
#            for k in range(n1):
#                for l in range(n2):
#                    tmpdiparray[k,l]=twoDdiparray[k,l]*\
#                                      pow(phase1,k)*pow(phase2,l)
#            [fftfreq1,fftfreq2,twodfreqarray]=twoDFFT(t1array,t2array,\
#                                                      tmpdiparray)
#            print("shape twodfreqarray "+str(shape(twodfreqarray)))
#            for k in range(n1):
#                for l in range(n2):
#                    retfftarray[k*npad1+i,l*npad2+j]=twodfreqarray[k,l]
#                    #retfftarray[mod(k*npad1+i+n1*npad1/2,n1*npad1),\
#                    #            mod(l*npad2+j+n2*npad2/2,n2*npad2)]=twodfreqarray[k,l]

    #retfftarray=fft.fftshift(retfftarray,axes=(0,1))
    #print("padfft fftfreq1\t"+str(fftfreq1))
    dw1=fftfreq1[1]-fftfreq1[0]
    dw2=fftfreq2[1]-fftfreq2[0]
    
    retfftfreq1=array(range(n1*npad1))*dw1/npad1+fftfreq1[0]
    retfftfreq2=array(range(n2*npad2))*dw2/npad2+fftfreq2[0]
    return [retfftfreq1,retfftfreq2,retfftarray]

##############################################
#First, read in phasematch_parameterkey.txt 

parameterarray=genfromtxt("phasematch_parameterkey.txt")
nt1=len(uniquevals(parameterarray[:,1]))#number of t1 values (FT over this)
nt2=len(uniquevals(parameterarray[:,2]))#number of t2 values
t1array=sort(uniquevals(parameterarray[:,1]))
t2array=sort(uniquevals(parameterarray[:,2]))

phasecol=5#column index for phase matching
fftcol=2
padarrays=True#True#interpolate FFT?



if(padarrays):
    npad1=4#4#interpolate
    npad2=1
else:
    npad1=1
    npad2=1

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

print("npad1, npad2 "+str(npad1)+"\t"+str(npad2))
####Perform 2D Fourier Transforms
if(fftcol==1):#fourier transform t1 column
    arraytofile(t1array,t3array,inpdiparray[:,0,:],"twoddiparray.dat")
    #[fftfreq1,fftfreq3,twodfreqarray]=FFT(t1array,t3array,inpdiparray[:,0,:])
    [fftfreq1,fftfreq3,twodfreqarray]=padFFT(t1array,t3array,inpdiparray[:,0,:],npad1,npad2)
    arraytofile(fftfreq1,fftfreq3,twodfreqarray,"twodfreqarray.dat")
if(fftcol==2):
    arraytofile(t2array,t3array,inpdiparray[0,:,:],"twoddiparray.dat")
    #[fftfreq2,fftfreq3,twodfreqarray]=FFT(t2array,t3array,inpdiparray[0,:,:])
    [fftfreq2,fftfreq3,twodfreqarray]=padFFT(t2array,t3array,inpdiparray[0,:,:],npad1,npad2)
    arraytofile(fftfreq2,fftfreq3,twodfreqarray,"twodfreqarray.dat")
    
