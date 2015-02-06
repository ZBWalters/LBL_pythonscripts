from scipy import *
from numpy import *
from pylab import *

Hnm=45.565# nm = Hnm/Estate
Hcm=2.1947e5
kboltz=3.16697647924e-6
aut=2.418884326505e-17
Hbyev=27.21
lightspeed=29979245800#cm/sec

temp=300.#300.#300.#300.
kbT=temp*kboltz
Gamma=kbT


#######vibrational information

lambda0m=[40,70]
omegaj=[207,244,312,372,438,514,718,813,938,1111,1450,1520,1790,2090]
sj=[.0013,.0072,.0450,.0578,.0450,.0924,.0761,.0578,.0313,.0578,.1013,.0265,.0072,.0013]

lambda0m=array(lambda0m)
omegaj=array(omegaj)
sj=array(sj)


def dos_vib(omega,kbT,xvals,yvals):
    xmax=max(xvals)
    harmonics=arange(omega/2,xmax*2,omega)
    print "harmonics",harmonics*Hcm
    print "kbT",kbT*Hcm
    for harm in harmonics:
        yvals=yvals+exp(-abs(xvals-harm)/(kbT))
        #plot(xvals,exp(-abs(xvals-harm)/(2*kbT)))
    return yvals

######Main program


dx=1/Hcm
xarray=arange(0,1000*dx,dx)
valarray=xarray*0.

for omega in omegaj:
    omegaH=omega/Hcm
    valarray=dos_vib(omegaH,kbT,xarray,valarray)

#figure()
plot(xarray*Hcm,valarray/max(valarray),'r-',label='300K')
xlabel('Vibrational Energy (cm$^{-1}$)')
ylabel('D$^{eff}(\omega)$')
legend(loc='best')
show()
