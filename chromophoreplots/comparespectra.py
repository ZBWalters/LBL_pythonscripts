from numpy import *
from scipy import *
from pylab import *

from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fmin

#Hnm=45.565# nm = Hnm/Estate
Hcm=2.1947e5
kboltz=3.16697647924e-6
aut=2.418884326505e-17
Hbyev=27.211396132
hplanck=4.1356668e-15#ev*sec
lightspeed=29979245800#cm/sec
Hnm=hplanck*lightspeed*1e7/Hbyev#~45.563 #wavelength in nm = Hnm/Estate

####helper functions
def maxindex(array):
    return list(array).index(max(array))

def rangerestrict(xmin,xmax,xarray,yarray):
    xret=[]
    yret=[]
    for i in range(len(xarray)):
        if((xmin<xarray[i]) and (xarray[i]<xmax)):
            xret.append(xarray[i])
            yret.append(yarray[i])
    xret=array(xret)
    yret=array(yret)
    return (xret,yret)

def weightedcenter(xarray,yarray):
    xysum=sum(xarray*abs(yarray)**2.)
    ysum=sum(abs(yarray)**2.)
    xavg=xysum/ysum
    return xavg


#paperfile="paper_absorption_300K.dat"
paperfile="paper_circulardichroism_300K.dat"
#paperfile="paper_fluorescence_300K.dat"

#paperfile="paper_absorption_77K.dat"
#paperfile="paper_fluorescence_77K.dat"

outputfile="sum.fft.dat"

paperarray=genfromtxt(paperfile)
papernm=paperarray[:,0]
paperI=paperarray[:,1]
paperE=Hnm/papernm
minE=min(paperE)
maxE=max(paperE)
print "minE,maxE",minE,maxE

paperE=paperE[-1:0:-1]
paperI=paperI[-1:0:-1]


outputarray=genfromtxt(outputfile)
outE=outputarray[:,0]
outI=outputarray[:,1]

(outE,outI)=rangerestrict(minE*Hcm-2000,maxE*Hcm+2000,outE,outI)


maxI=max(abs(outI))

print "energy offset (peaks)",outE[maxindex(outI)]-paperE[maxindex(paperI)]*Hcm
print "energy offset",-weightedcenter(outE,outI)+weightedcenter(paperE*Hcm,paperI)
figure()
plot(paperE*Hcm,paperI/max(abs(paperI)),'r-',label='Experiment')
plot(outE-outE[maxindex(outI)]+paperE[maxindex(paperI)]*Hcm,outI/maxI,'k-',label='Theory')
#plot(outE-weightedcenter(outE,outI)+weightedcenter(paperE*Hcm,paperI),outI/maxI,label='output')
plot(outE,outI/maxI,'k:',label='Theory (unshifted)')
xlim(minE*Hcm,maxE*Hcm)
xlabel('Frequency (cm$^{-1}$)')
legend(loc='best')
show()




