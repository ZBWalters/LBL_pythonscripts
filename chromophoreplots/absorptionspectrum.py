from numpy import *
from scipy import *
from pylab import *

aut=24.18e-18
Hcm=2.1947e5
Hbyev=27.21

file1="./fort.13"

def calcfft(filename):
    dipvst=genfromtxt(filename)
    timearray=dipvst[:,0]
    diparray=dipvst[:,1]+dipvst[:,2]*1j
    deltat=timearray[1]-timearray[0]
    (nrow,ncol)=shape(dipvst)
    frequencies=fftfreq(nrow,d=deltat)
    frequencies=frequencies*aut*Hcm
    frequencies=fftshift(frequencies)*2*pi
    fftdiparray=fft(diparray)
    fftdiparray=fftshift(fftdiparray)
    return (frequencies,fftdiparray)

def printfft(filename):
    (frequencies,fftdiparray)=calcfft(filename)
    outfilename=filename+".fft.dat"
    outfile=open(outfilename,'w')
    for i in range(len(frequencies)):
        outfile.write(str(frequencies[i])+"\t"+str(real(fftdiparray[i]))+"\t"+str(real(fftdiparray[i]))+"\n")
    outfile.close()
    

def plotfft(filename):
    figure()
    dipvst=genfromtxt(filename)
    
    timearray=dipvst[:,0]

    (nrow,ncol)=shape(dipvst)
    print "nrow,ncol",nrow,ncol

    diparray=dipvst[:,1]+dipvst[:,2]*1j

    deltat=timearray[1]-timearray[0]
    frequencies=fftfreq(nrow,d=deltat)
    frequencies=frequencies*aut*Hcm
    frequencies=fftshift(frequencies)*2*pi
    fftdiparray=fft(diparray)
    fftdiparray=fftshift(fftdiparray)
    #plot(45.565/(frequencies/Hcm),real(fftdiparray))
    plot((frequencies),real(fftdiparray))
    xlim((16000,22000))#xlim((450,650))#xlim((16000,22000))
    title(filename)
    show()



filelist=["./fort.21","./fort.22","./fort.23"]
for filename in filelist:
    plotfft(filename)
    printfft(filename)

(frequencies,fftx)=calcfft(filelist[0])
(frequencies,ffty)=calcfft(filelist[1])
(frequencies,fftz)=calcfft(filelist[2])

figure()
#plot(45.565/(frequencies/Hcm),fftx+ffty+fftz)
plot((frequencies),fftx+ffty+fftz)
xlim(16000,22000)#xlim(450,650)#xlim(16000,22000)
title("sum")
show()

fftsum=(fftx+ffty+fftz)*frequencies
fftsum2=(real(fftx)**2.+real(ffty)**2.+real(fftz)**2.)*frequencies**3.
outfilename="sum.fft.dat"
outfile=open(outfilename,'w')
for i in range(len(frequencies)):
#    outfile.write(str(45.565/(frequencies[i]/Hcm))+"\t"+str(real(fftsum[i]))+"\t"+str(imag(fftsum[i]))+"\n")
    outfile.write(str((frequencies[i]))+"\t"+str(real(fftsum[i]))+"\t"+str(imag(fftsum[i]))+"\n")
outfile.close()

outfilename="sum2.fft.dat"
outfile=open(outfilename,'w')
for i in range(len(frequencies)):
#    outfile.write(str(45.565/(frequencies[i]/Hcm))+"\t"+str(real(fftsum2[i]))+"\t"+str(imag(fftsum2[i]))+"\n")
    outfile.write(str((frequencies[i]))+"\t"+str(real(fftsum2[i]))+"\t"+str(imag(fftsum2[i]))+"\n")
outfile.close()


