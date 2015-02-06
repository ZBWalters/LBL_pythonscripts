import sys
sys.path.append(r'/Users/zwalters/pythonscripts')
from makefigs import *

def nm_to_eV(nm):
    hc=1239.841#planck's constant in eV*nm
    eV=hc/nm
    return eV

def contourcolorlist(twopower):
    maxval=1
    minval=0.1
    delta=(maxval-minval)/twopower
    return list(map(str,arange(minval,maxval+delta,delta)[1:]))#'w'

contourcmapname="cubehelix_r"#'CMRmap_r'#'gnuplot2_r'#'cubehelix_r'#'winter'
densitycontourcmapname='gray'
contourcmap=plt.cm.get_cmap(contourcmapname)
contourcmap2=plt.cm.get_cmap(densitycontourcmapname)
twopower=14
range=pow(2,twopower)
ncontours=twopower
showcontours=True
contouralpha=1.
contourlinewidths=1.5
contourcolors=contourcolorlist(twopower)
savefigs=True
evlo=0
evhi=40
nlo=None
nhi=None
prefix1='logrange_2_'+str(twopower)+'_'

makecontourfigs(prefix=prefix1, logrange=log(range), 
                savefigs=savefigs,
                evlo=evlo, evhi=evhi, nlo=nlo, nhi=nhi, 
                ncontours=ncontours, contouralpha=contouralpha, densityalpha=0,
                contourcolors=contourcolors, 
                contourlinewidths=contourlinewidths,
                contourcmapname=contourcmapname, backgroundcolor="white") 
prefix2='linear_zeroinitialphase_'
plot_time_phase(colorpower=1, savefigs=savefigs, prefix=prefix2,
                showcontours=showcontours, ncontours=ncontours,
                contouralpha=contouralpha,
                contourcmapname=contourcmapname, contourlinewidths=contourlinewidths,
                backgroundcolor="white")

prefix3=prefix1+"zeroinitialphase_"
plot_freq_phase(colorpower='log', logrange=log(range), prefix=prefix3,
                savefigs=savefigs, ylo=evlo, yhi=evhi,
                showcontours=showcontours, ncontours=ncontours,
                contouralpha=contouralpha, 
                contourcmapname=contourcmapname, contourlinewidths=contourlinewidths,
                backgroundcolor='white')

prefix4=prefix1+"removedrivingphase_"
adjust_kIR_vs_t_plot(wpr=25, ws=nm_to_eV(760), logrange=log(range), nlo=nlo, nhi=nhi,
                     evlo=evlo, evhi=evhi, prefix=prefix4,
                     savefigs=savefigs,showcontours=showcontours, ncontours=ncontours,
                     contourcolors=contourcolors, contourcmapname=contourcmapname,
                     contourlinewidths=contourlinewidths)
plt.show()
