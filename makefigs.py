from twoDplot import *
from numpy.fft import *
#use the tools developed in twoDplot.py to plot figures from optical streaking



##################################################
#useful for plotting 2D spectra
twoDlabellist=["","$k_{1}$","$k_{2}$","$k_{3}$","$k_{1}-k_{2}+k_{3}$","$k_{1}+k_{2}-k_{3}$","$-k1+k2+k3$","$2k_{1}-k_{2}$","$2k_{1}-k_{3}$","$-k_{1}+2k_{2}$","$2k_{2}-k_{3}$","$-k_{1}+2k_{3}$","$-k_{2}+2k_{3}$"]

def dipfilename(i):
    return "twoddiparray."+str(i)+".dat"

def freqfilename(i):
    return "twodfreqarray."+str(i)+".dat"

def shiftfreqfilename(i):
    return "twodfreqarray."+str(i)+"_shift.dat"

################################################################################
#Modifications to the output graphs which illustrate a particular point
def plot_freq_phase(dirstr="1/", colorpower='log', logrange=6, legend="",
                    xlabel="$\phi$/$\pi$",ylabel="$\omega_{out}$ (eV)",
                    theme='hls',  xmultfactor=1., ymultfactor=Hrt, xlo=None,
                    xhi=None,  ylo=None, yhi=None, ymasklo=None,ymaskhi=None,
                    inpfontsize=20, savefigs=False, prefix="",
                    figfilename="omega_vs_phi_normalized_phase.png",
                    showcontours=False, ncontours=5, contouralpha=0.5,
                    contourlinewidths=1, contourcmapname='spectral',
                    backgroundcolor='white'):
    filename=dirstr+"phase_vs_freq.dat"
    [xdat,ydat,zdat]=datfromfile(filename)
    xarray,yarray,zarray=dattoarrays(xdat,ydat,zdat)
    yvec=yarray[0,:]
    (n0,n1)=shape(zarray)
    if(ymasklo!=None and ymaskhi!=None):
        for i in range(n1):
            if(yvec[i]*ymultfactor>ymasklo and yvec[i]*ymultfactor<ymaskhi):
                zarray[:,i]=0.
    for i in range(n1):
        zarray[:,i]/=exp(1j*angle(zarray[0,i]))
#    fig=imshowplot_hsv(xarray, yarray, zarray, colorpower, logrange, legend,
#                       xlabel, ylabel, theme, xmultfactor, ymultfactor, xlo,
#                       xhi, ylo, yhi, inpfontsize, showcontours=showcontours,
#                       ncontours=ncontours, contouralpha=contouralpha,
#                       contourcolors=contourcolors, contourcmap=contourcmap,
#                       contourlinewidths=contourlinewidths)
    fig=contourplot(xarray, yarray, zarray, colorpower, logrange, legend,
                    xlabel, ylabel, theme, xmultfactor, ymultfactor, xlo, xhi,
                    ylo, yhi, inpfontsize,
                    ncontours=ncontours, contouralpha=contouralpha,
                    contourlinewidths=contourlinewidths,
                    contourcmapname=contourcmapname,
                    backgroundcolor=backgroundcolor)
 
    if(savefigs):
        fig.savefig(prefix+figfilename)
    return fig


def plot_time_phase(dirstr="1/", colorpower='log', logrange=6, legend="",
                    xlabel="$\phi$/$\pi$",ylabel="$t-t_{start}$ (fs)",
                    theme='hls',  xmultfactor=1., ymultfactor=aut/1000.,
                    xlo=None, xhi=None,  ylo=None, yhi=None,
                    ymasklo=None,ymaskhi=None, savefigs=False, prefix="",
                    figfilename="t_vs_phi_normalized_phase.png",
                    inpfontsize=20, showcontours=False, ncontours=5,
                    contouralpha=0.5, contourcmapname='spectral',
                    contourlinewidths=1, backgroundcolor='white'):
    filename=dirstr+"phase_vs_t.dat"
    [xdat,ydat,zdat]=datfromfile(filename)
    xarray,yarray,zarray=dattoarrays(xdat,ydat,zdat)
    yvec=yarray[0,:]
    (n0,n1)=shape(zarray)
    if(ymasklo!=None and ymaskhi!=None):
        for i in range(n1):
            if(yvec[i]*ymultfactor>ymasklo and yvec[i]*ymultfactor<ymaskhi):
                zarray[:,i]=0.
    for i in range(n1):
        zarray[:,i]/=exp(1j*angle(zarray[0,i]))
#    fig=imshowplot_hsv(xarray, yarray, zarray, colorpower, logrange, legend,
#                       xlabel, ylabel, theme, xmultfactor, ymultfactor, xlo,
#                       xhi, ylo, yhi, inpfontsize, showcontours=showcontours,
#                       ncontours=ncontours, contouralpha=contouralpha,
#                       contourcolors=contourcolors, contourcmap=contourcmap,
#                       contourlinewidths=contourlinewidths)
    fig=contourplot(xarray, yarray, zarray, colorpower, logrange, legend,
                    xlabel, ylabel, theme, xmultfactor, ymultfactor, xlo, xhi,
                    ylo, yhi, inpfontsize, ncontours=ncontours,
                    contourlinewidths=contourlinewidths,
                    backgroundcolor=backgroundcolor,
                    contourcmapname=contourcmapname)
 
    if(savefigs):
        fig.savefig(prefix+figfilename)
    return fig

###############

def makecontourfigs(prefix="", dirstr="1/", savefigs=False, nlo=None, nhi=None,
                    evlo=None, evhi=None, **kwargs):
    fig1=contourplotfile(filename=dirstr+"kIR_vs_t.dat", xlo=nlo, xhi=nhi,
                         ymultfactor=aut/1000, colorpower='log', **kwargs)
    fig2=contourplotfile(filename=dirstr+"kIR_vs_freq.dat", xlo=nlo, xhi=nhi,
                         ymultfactor=Hrt, ylo=evlo, yhi=evhi, colorpower='log', **kwargs)
    fig3=contourplotfile(filename=dirstr+"phase_vs_t.dat", colorpower=1,
                         ymultfactor=aut/1000,  **kwargs)
    fig4=contourplotfile(filename=dirstr+"phase_vs_freq.dat", colorpower='log',
                         ymultfactor=Hrt, ylo=evlo, yhi=evhi, **kwargs)
    if(savefigs):
        fig1.savefig(prefix+"kIR_vs_t_contour.png")
        fig2.savefig(prefix+"kIR_vs_freq_contour.png")
        fig3.savefig(prefix+"phase_vs_t_contour.png")
        fig4.savefig(prefix+"phase_vs_freq_contour.png")
        

#def makecontourfigs(colorpower='log', logrange=6, hival=60, dirstr="1/", prefix="",
#             savefigs=False, evlo=None, evhi=None, nlo=None, nhi=None,
#             theme='hls', inpfontsize=20, showcontours=False, ncontours=5,
#             contouralpha=1., densityalpha=1., contourcolors='w', contourcmapname=None, contourlinewidths=1):
#    
#    fig1=contourplotfile(filename=dirstr+"kIR_vs_t.dat",colorpower,logrange,\
#                        legend="",xlabel="$n_{s}$",
#                        ylabel=("t (fs)"),ymultfactor=aut/1000.,
#                        inpfontsize=inpfontsize,xlo=nlo,xhi=nhi,theme=theme,
#                        ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmapname=contourcmapname,
#                        contourlinewidths=contourlinewidths)
#    fig2=contourplotfile(filename=dirstr+"kIR_vs_freq.dat", colorpower,logrange,\
#                        legend="",
#                        xlabel="$n_{s}$",ylabel="$\omega_{out}$ (eV)",
#                        ymultfactor=Hrt,inpfontsize=inpfontsize, xlo=nlo, xhi=nhi,
#                        ylo=evlo,yhi=evhi,theme=theme, 
#                        ncontours=ncontours, contouralpha=contouralpha,
#                        densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmapname=contourcmapname,
#                        contourlinewidths=contourlinewidths)
#    fig3=contourplotfile(filename=dirstr+"phase_vs_t.dat",colorpower=1.,logrange=logrange,\
#                        legend="", xlabel="$\phi$/$\pi$", ylabel="t (fs)",
#                        ymultfactor=aut/1000., inpfontsize=inpfontsize,
#                        theme=theme,    ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmapname=contourcmapname,
#                        contourlinewidths=contourlinewidths)
#    fig4=contourplotfile(filename=dirstr+"phase_vs_freq.dat",colorpower,logrange,\
#                        legend="", xlabel="$\phi$/$\pi$",ylabel="$\omega_{out}$ (eV)",
#                        ymultfactor=Hrt, inpfontsize=inpfontsize,
#                        ylo=evlo, yhi=evhi, theme=theme,
#                        ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=1.,
#                        contourcolors=contourcolors, contourcmapname=contourcmapname,
#                        contourlinewidths=contourlinewidths)
#################################################################################
#    fig1=contourplotfile(dirstr+"kIR_vs_t.dat",colorpower,logrange,\
#                        legend="",xlabel="$n_{s}$",
#                        ylabel=("t (fs)"),ymultfactor=aut/1000.,
#                        inpfontsize=inpfontsize,xlo=nlo,xhi=nhi,theme=theme,
#                        ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmap=contourcmap)
#
#    fig2=contourplotfile(dirstr+"kIR_vs_freq.dat",colorpower,logrange,\
#                        legend="",xlabel="$n_{s}$",
#                        ylabel=("t (fs)"),ymultfactor=aut/1000.,
#                        inpfontsize=inpfontsize,xlo=nlo,xhi=nhi,theme=theme,
#                        ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmap=contourcmap)
#
#    fig3=contourplotfile(dirstr+"phase_vs_t.dat",colorpower,logrange,\
#                        legend="",xlabel="$n_{s}$",
#                        ylabel=("t (fs)"),ymultfactor=aut/1000.,
#                        inpfontsize=inpfontsize,xlo=nlo,xhi=nhi,theme=theme,
#                        ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmap=contourcmap)
#
#    fig4=contourplotfile(dirstr+"phase_vs_freq.dat",colorpower,logrange,\
#                        legend="",xlabel="$n_{s}$",
#                        ylabel=("t (fs)"),ymultfactor=aut/1000.,
#                        inpfontsize=inpfontsize,xlo=nlo,xhi=nhi,theme=theme,
#                        ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmap=contourcmap)
#

def makefigs(dirstr="1/", savefigs=False, evlo=None, evhi=None,nlo=None,
             nhi=None, **kwargs):
    fig1=imshowplotfile(dirstr+"kIR_vs_t.dat", ylabel="t (fs)", xlabel="$n_{s}$",
                        ymultfactor=aut/1000., xlo=nlo, xhi=nhi, **kwargs)
    fig2=imshowplotfile(dirstr+"kIR_vs_freq.dat", ylabel="$\omega_{out}$ (eV)",
                        xlabel="$n_{s}$", ymultfactor=Hrt, xlo=nlo, xhi=nhi,
                        ylo=evlo, yhi=evhi, **kwargs)
    fig3=imshowplotfile(dirstr+"phase_vs_t.dat", xlabel="$\phi$/$\pi$",
                        ylabel="t (fs)", ymultfactor=aut/1000., **kwargs) 
    fig4=imshowplotfile(dirstr+"phase_vs_freq.dat", xlabel="$\phi$/$\pi$",
                         ylabel="$\omega_{out}$ (eV)", ymultfactor=Hrt,
                         ylo=evlo, yhi=evhi, arrayfunction=logz, **kwargs)
    if(savefigs):
        fig1.savefig(prefix+"kIR_vs_t.png")
        fig2.savefig(prefix+"kIR_vs_freq.png")
        fig3.savefig(prefix+"phase_vs_t.png")
        fig4.savefig(prefix+"phase_vs_freq.png")


#def makefigs(colorpower='log', logrange=6, hival=60, dirstr="1/", prefix="",
#             savefigs=False, evlo=None, evhi=None, nlo=None, nhi=None,
#             theme='hls', inpfontsize=20, showcontours=False, ncontours=5,
#             contouralpha=1., densityalpha=1., contourcolors='w', contourcmap=None, contourlinewidths=1):
#    
#    fig1=imshowplotfile(dirstr+"kIR_vs_t.dat",colorpower,logrange,\
#                        legend="",xlabel="$n_{s}$",
#                        ylabel=("t (fs)"),ymultfactor=aut/1000.,
#                        inpfontsize=inpfontsize,xlo=nlo,xhi=nhi,theme=theme,
#                        showcontours=showcontours, ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmap=contourcmap, contourlinewidths=contourlinewidths)
#    fig2=imshowplotfile(dirstr+"kIR_vs_freq.dat", colorpower,logrange,\
#                        legend="",
#                        xlabel="$n_{s}$",ylabel="$\omega_{out}$ (eV)",
#                        ymultfactor=Hrt,inpfontsize=inpfontsize, xlo=nlo, xhi=nhi,
#                        ylo=evlo,yhi=evhi,theme=theme, showcontours=showcontours,
#                        ncontours=ncontours, contouralpha=contouralpha,
#                        densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmap=contourcmap)
#    fig3=imshowplotfile(dirstr+"phase_vs_t.dat",colorpower=1.,logrange=logrange,\
#                        legend="", xlabel="$\phi$/$\pi$", ylabel="t (fs)",
#                        ymultfactor=aut/1000., inpfontsize=inpfontsize,
#                        theme=theme,  showcontours=showcontours,  ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=densityalpha,
#                        contourcolors=contourcolors, contourcmap=contourcmap,
#                        contourlinewidths=contourlinewidths)
#    fig4=imshowplotfile(dirstr+"phase_vs_freq.dat",colorpower,logrange,\
#                        legend="", xlabel="$\phi$/$\pi$",ylabel="$\omega_{out}$ (eV)",
#                        ymultfactor=Hrt, inpfontsize=inpfontsize,
#                        ylo=evlo, yhi=evhi, arrayfunction=logz, theme=theme,
#                        showcontours=showcontours, ncontours=ncontours,
#                        contouralpha=contouralpha, densityalpha=1.,
#                        contourcolors=contourcolors, contourcmap=contourcmap,
#                        contourlinewidths=contourlinewidths)
#    if(savefigs):
#        fig1.savefig(prefix+"kIR_vs_t.png")
#        fig2.savefig(prefix+"kIR_vs_freq.png")
#        fig3.savefig(prefix+"phase_vs_t.png")
#        fig4.savefig(prefix+"phase_vs_freq.png")
#

def makefigset(savebool=False,inplogrange=6):
    makefigs(colorpower='log',logrange=inplogrange,dirstr="1/",prefix="nprobe_1_",savefigs=savebool)
    makefigs(colorpower='log',logrange=inplogrange,dirstr="0/",prefix="nprobe_0_",savefigs=savebool)

def dirmakefigset(dirname,logrange=6,showfigs=True,savefigs=False):
    currentdir=os.getcwd()
    os.chdir(dirname)
    print("directory "+dirname)
    makefigset(savefigs,logrange)
    if(showfigs):
        plt.show()
    else:
        plt.close('all')#clear figures
    os.chdir(currentdir)

def dirlistmakefigset(dirlist,logrange=6,showfigs=True,savefigs=False):
    for dirname in dirlist:
        dirmakefigset(dirname,logrange,showfigs,savefigs)

def adjust_kIR_vs_t_plot(wpr, ws, colorpower='log', logrange=6, hival=60,
                         dirstr="1/", prefix="", savefigs=False, evlo=None,
                         evhi=None, nlo=None, nhi=None,theme='hls',
                         inpfontsize=20, showcontours=False, ncontours=10,
                         contourcolors='w', contourcmapname=None, contourlinewidths=1):
    phiarray,tarray,diparray=arraysfromfile(dirstr+"phase_vs_t.dat")
    narray,tarray,diparray=arraysfromfile(dirstr+"kIR_vs_t.dat")
    
#   Remove the rapidly oscillating components due to npr*wpr and ns*ws
    #diparray*=exp(-1j*(wpr/Hrt)*tarray)#consider leaving in ns*ws*t phase
    diparray*=exp(-1j*(ws/Hrt*narray)*tarray)
#    fig1=imshowplot_hsv(xarray=narray, yarray=tarray, zarray=diparray,
#                        xlabel="$n_{s}$", ylabel="$t-t_{start}$ (fs)",
#                        ymultfactor=aut/1000, colorpower=colorpower,
#                        logrange=logrange, xlo=nlo, xhi=nhi, theme=theme,
#                        inpfontsize=inpfontsize, showcontours=showcontours,
#                        ncontours=ncontours, contourcolors=contourcolors,
#                        contourcmap=contourcmap,
#                        contourlinewidths=contourlinewidths)
    fig1=contourplot(xarray=narray, yarray=tarray, zarray=diparray,
                        xlabel="$n_{s}$", ylabel="$t-t_{start}$ (fs)",
                        ymultfactor=aut/1000, colorpower=colorpower,
                        logrange=logrange, xlo=nlo, xhi=nhi, theme=theme,
                        inpfontsize=inpfontsize,
                        contourcmapname=contourcmapname, ncontours=ncontours,
                        contourlinewidths=contourlinewidths, backgroundcolor='white')
#
#   Now take fourier transform of diparray(n,t) to get diparray(phi,t)
    diparraycopy=fftshift(diparray,axes=[0])
    iftdiparray=ifft(diparraycopy,axis=0)
    #plot the slowly varying time vs phase plot
#    fig2=imshowplot_hsv(xarray=phiarray, yarray=tarray, zarray=iftdiparray,
#                        xlabel="$\phi/\pi$", ymultfactor=aut/1000,
#                        ylabel="$t-t_{start}$ (fs)", colorpower=colorpower,
#                        logrange=logrange, theme=theme,
#                        inpfontsize=inpfontsize, showcontours=showcontours,
#                        ncontours=ncontours, contourcolors=contourcolors,
#                        contourcmap=contourcmap, contourlinewidths=contourlinewidths)
    fig2=contourplot(xarray=phiarray, yarray=tarray, zarray=iftdiparray,
                     xlabel="$\phi/\pi$", ymultfactor=aut/1000,
                     ylabel="$t-t_{start}$ (fs)", colorpower=colorpower,
                     logrange=logrange, theme=theme, inpfontsize=inpfontsize,
                     contourcmapname=contourcmapname, ncontours=ncontours,
                     contourlinewidths=contourlinewidths, backgroundcolor='white')
#
#   take the fourier transform over time to get the slowly varying w vs phi plot
    wftdiparray=fft(diparray,axis=1)
    wftdiparray=fftshift(wftdiparray,axes=[1])
    warray=warrayfromtarray(tarray)
#    fig3=imshowplot_hsv(xarray=narray, yarray=warray, zarray=wftdiparray,
#                        colorpower=colorpower, logrange=logrange, theme=theme,
#                        xlabel="$n_{s}$",
#                        ylabel="$\omega_{out}-\omega_{probe}-n_{s}\omega_{s}$ (eV)",
#                        xlo=nlo, xhi=nhi, ylo=evlo, yhi=evhi,
#                        ymultfactor=Hrt, inpfontsize=inpfontsize,
#                        showcontours=showcontours, ncontours=ncontours,
#                        contourcolors=contourcolors, contourcmap=contourcmap,
#                        contourlinewidths=contourlinewidths)
    fig3=contourplot(xarray=narray, yarray=warray, zarray=wftdiparray,
                     colorpower=colorpower, logrange=logrange, theme=theme,
                     xlabel="$n_{s}$",
                     ylabel="$\omega_{out}-n_{s}\omega_{s}$ (eV)",
                     xlo=nlo, xhi=nhi, ylo=evlo, yhi=evhi,
                     ymultfactor=Hrt, inpfontsize=inpfontsize,
                     contourcmapname=contourcmapname, ncontours=ncontours,
                     contourlinewidths=contourlinewidths, backgroundcolor='white')
#
#   take the fourier transform over ns to get dw vs phi plot
    dwphiarray=fftshift(wftdiparray,axes=[0])
    dwphiarray=ifftn(dwphiarray,axes=[0])
#    fig4=imshowplot_hsv(xarray=phiarray, yarray=warray, zarray=dwphiarray,
#                        colorpower=colorpower, logrange=logrange, theme=theme,
#                        xlabel="$\phi/\pi$",
#                        ylabel="$\omega_{out}-\omega_{probe}-n_{s}\omega_{s}$ (eV)",
#                        ylo=evlo, yhi=evhi, ymultfactor=Hrt,
#                        inpfontsize=inpfontsize, showcontours=showcontours,
#                        ncontours=ncontours, contourcolors=contourcolors, contourcmap=contourcmap, contourlinewidths=contourlinewidths)
    fig4=contourplot(xarray=phiarray, yarray=warray, zarray=dwphiarray,
                        colorpower=colorpower, logrange=logrange, theme=theme,
                        xlabel="$\phi/\pi$",
                        ylabel="$\omega_{out}-\omega_{probe}-n_{s}\omega_{s}$ (eV)",
                        ylo=evlo, yhi=evhi, ymultfactor=Hrt,
                        inpfontsize=inpfontsize,
                        contourcmapname=contourcmapname, ncontours=ncontours,
                        contourlinewidths=contourlinewidths, backgroundcolor='white')

#    fig4p=imshowplot(xarray=phiarray, yarray=warray, zarray=abs(dwphiarray),
#                        colorpower='log', logrange=logrange,cmap="cubehelix_r",
#                        xlabel="$\phi/\pi$",
#                        ylabel="$\omega_{out}-\omega_{probe}-n_{s}\omega_{s}$ (eV)",
#                        ylo=evlo, yhi=evhi, ymultfactor=Hrt,
#                        inpfontsize=inpfontsize, symmetricrange=False) 
 

##   

    if(savefigs):
            fig1.savefig(prefix+"kIR_vs_t.png")
            fig2.savefig(prefix+"phi_vs_t.png")
            fig3.savefig(prefix+"kIR_vs_w.png")
            fig4.savefig(prefix+"phi_vs_w.png")
     
def warrayfromtarray(tarray):
    tvec=tarray[0,:]
    shp=shape(tarray)
    wvec=array(sorted(fftfreq(len(tvec),tvec[1]-tvec[0])))*2*pi
    retarray=zeros(shp)
    for i in range(shp[0]):
        retarray[i,:]=wvec
    return retarray

def adjust_w_vs_phi_plot(wpr, colorpower='log', logrange=6, hival=60,
                         dirstr="1/", prefix="", savefigs=False,theme='hls'):
    [phidat,tdat,dipdat]=datfromfile(dirstr+"phase_vs_t.dat")
    phiarray,tarray,diparray=dattoarrays(phidat,tdat,dipdat)
    diparray*=exp(-1j*(wpr/Hrt)*tarray)
    ftdiparray=fft(diparray,axis=1)
    ftdiparray=fftshift(ftdiparray,axes=[1])
    warray=warrayfromtarray(tarray)
    fig1=imshowplot_hsv(phiarray, warray, ftdiparray, colorpower, logrange,
                        theme=theme, ymultfactor=Hrt)
    fig1a=imshowplot_fun(phiarray, warray, ftdiparray, colorpower, logrange,
                        theme=theme, ymultfactor=Hrt,arrayfunction=logz)

def makeadjustedfigs(wpr, ws, colorpower='log', logrange=6, hival=60,
                     dirstr="1/", prefix="", savefigs=False,theme='hls'):
    adjust_kIR_vs_t_plot(wpr, ws, colorpower, logrange, hival,
                         dirstr, prefix, savefigs,theme)
    adjust_w_vs_phi_plot(wpr, colorpower, logrange, hival,
                         dirstr, prefix, savefigs,theme)
################################################################################    

#first try
#def nDFFT(canglediparray,axislist, shiftaxes=False, inv=False):
#    if(inv):
#        fftdiparray=ifftn(canglediparray,axes=axislist)
#    else:
#        fftdiparray=fftn(canglediparray,axes=axislist)
#    if(shiftaxes):
#        if(inv):
#            fftdiparray=ifftshift(fftdiparray,axes=axislist)
#        else:
#            fftdiparray=fftshift(fftdiparray,axes=axislist)
#    return fftdiparray
#

def nDFFT(cinparray,axislist,shiftaxes=False, inv=False):
    if(inv and shiftaxes):
        tmpcinparray=ifftshift(cinparray,axes=axislist)
        fftdiparray=ifftn(tmpcinparray,axes=axislist)
        fftdiparray=ifftshift(fftdiparray,axes=axislist)
    if(not(inv) and shiftaxes):
        fftdiparray=fftn(cinparray,axes=axislist)
        fftdiparray=fftshift(fftdiparray,axes=axislist)
    if(not(inv) and not(shiftaxes)):
        fftdiparray=fftn(cinparray,axes=axislist)
    if(inv and not(shiftaxes)):
        fftdiparray=ifftn(cinparray,axes=axislist)
    return fftdiparray


def subtractzerovec(inparray):
    zerovec=inparray[0,:]
    retarray=copy(inparray)
    for i in range(shape(inparray)[0]):
        retarray[i,:]-=zerovec
    return retarray
######################################
#adjust_kIR_vs_t_plot(20,.0565*Hrt,logrange=4)
#plt.show()
