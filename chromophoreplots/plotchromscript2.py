from numpy import *
from pylab import *
import glob


def plotfiles(file1,file2,filename):
    array1=genfromtxt(file1)
    array2=genfromtxt(file2)

    clrs=['black','red','green','blue','orange','brown','grey','violet']

#new ordering
    names=['t',r'$\beta_{50C}$',r'$\alpha_{19A}$',r'$\alpha_{19B}$',r'$\beta_{82C}$',r'$\beta_{158C}$',r'$\beta_{50D}$',r'$\beta_{82D}$',r'$\beta_{158D}$']

#old ordering
#names=['t','free',r'$\alpha_{19A}$',r'$\alpha_{19B}$',r'$\beta_{50C}$',r'$\beta_{82C}$',r'$\beta_{158C}$',r'$\beta_{50D}$',r'$\beta_{82D}$',r'$\beta_{158D}$']

    (n1,n2)=shape(array1)
    for i in range(1,n2):
        plot(array1[:,0]*1e12,array1[:,i],ls="-",c=clrs[i-1],label=names[i])
        plot(array2[:,0]*1e12,array2[:,i],ls=":",c=clrs[i-1])
    xlabel('Time (ps)')
    legend(loc='best')
    #savefig(filename)
    show()
    return

#main program
filelist1=sort(glob.glob("./fort.*01"))
filelist2=sort(glob.glob("./expdecay*.dat"))

for i in range(len(filelist1)):
    figure()
    filename="pops_vs_time_p"+str(i+1)+str(i+1)+"_T77.eps"
    plotfiles(filelist1[i],filelist2[i],filename)
