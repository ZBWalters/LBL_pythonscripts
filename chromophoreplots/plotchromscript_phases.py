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
    #show()
    return

def plotfile(file1,linestyle,showlabels):
    array1=genfromtxt(file1)
    clrs=['black','red','green','blue','orange','brown','grey','violet']

#new ordering
    names=['t',r'$\beta_{50C}$',r'$\alpha_{19A}$',r'$\alpha_{19B}$',r'$\beta_{82C}$',r'$\beta_{158C}$',r'$\beta_{50D}$',r'$\beta_{82D}$',r'$\beta_{158D}$']
#old ordering
#names=['t','free',r'$\alpha_{19A}$',r'$\alpha_{19B}$',r'$\beta_{50C}$',r'$\beta_{82C}$',r'$\beta_{158C}$',r'$\beta_{50D}$',r'$\beta_{82D}$',r'$\beta_{158D}$']
    (n1,n2)=shape(array1)
    for i in range(1,n2):
        if(showlabels):
            plot(array1[:,0]*1e12,array1[:,i],ls=linestyle,c=clrs[i-1],label=names[i])
        else:
           plot(array1[:,0]*1e12,array1[:,i],ls=linestyle,c=clrs[i-1]) 
    

    
#main program
filelist1=sort(glob.glob("./fort.10*1"))
lslist=["-","--","-.",":"]

figure()
plotfile(filelist1[0],lslist[0],True)
for i in range(1,len(filelist1)):
    plotfile(filelist1[i],lslist[i],False)
xlabel("Time (ps)")
legend(loc="best")
show()
