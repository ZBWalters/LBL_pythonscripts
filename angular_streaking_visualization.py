from numpy import *
import subprocess
import os
from mayavi import mlab
from itertools import product as iterproduct
from scipy.special import sph_harm
from scipy.interpolate import interp1d
from numba import jit, autojit
from tvtk.util.ctf import ColorTransferFunction
from tvtk.util.ctf import PiecewiseFunction

################################################################################
#helper functions
def ftindxlist(inplist):
    retarray=[]
    for i in range(len(inplist)):
        retarray.append(ftveclist[i].index(inplist[i]))
    return retarray

def nvec(n):
    nmin=-int(floor(n/2.))
    return range(nmin,nmin+n)

def arrayouter(inparray,inpvec):
    retarray=outer(ndarray.flatten(inparray),inpvec)
    (n0,n1)=shape(inparray)
    n2=len(inpvec)
    retarray=retarray.reshape(n0,n1,n2)
    return retarray

def makepolararrays(ntheta=100,nphi=100):
    tmpthetaarray, tmpphiarray=thetaphigrid(ntheta,nphi)
    nt=len(veclist[list(namelist).index("tvec")])
    thetaarray=arrayouter(tmpthetaarray,ones(nt/2))
    phiarray=arrayouter(tmpphiarray,ones(nt/2))
    rarray=arrayouter(ones(shape(tmpthetaarray)),freqvec[(nt/2):])
    return thetaarray, phiarray, rarray

def makecartesianarrays(xlo=-1, xhi=1, ylo=-1, yhi=1, zlo=-1, zhi=1, nx=10,
                        ny=10, nz=10, **kwargs):
    x,y,z=mgrid[xlo:xhi:1j*nx, ylo:yhi:1j*ny, zlo:zhi:1j*nz]
    return x,y,z


def polartoxyz(theta,phi,r):
    x=r*sin(theta)*cos(phi)
    y=r*sin(theta)*sin(phi)
    z=r*cos(theta)
    return x,y,z

def xyztopolar(x,y,z):
    tiny=1.e-6
    r=sqrt(x**2+y**2+z**2)
    theta=arccos(z/(r+tiny))
    phi=angle(x+1j*y)
    return r, theta, phi


def arraytoint(inparray):
    inpshape=shape(inparray)
    return array(map(int,inparray.flatten())).reshape(inpshape)



def onedinterp(xvals, xinterp, yinterp):
    interpfun=interp1d(xinterp,yinterp)
    inpshape=shape(xvals)
#mask array tells us that we're out of the interpolation region
#trying to interpolate at these points will give bad results
    mask=(xvals>=xinterp[0])*(xvals<=xinterp[-1])
#first, we will replace these values with dummy input values...
    evalpts=where(mask,xvals,xinterp[0])
#then evaluate the interpolation on the new set of points
    retarray=interpfun(evalpts)
#and finally, replace the interpolated value at the dummy points with 0
    retarray=where(mask,retarray,0)
    return retarray
    


################################################################################
#process arrays
def dipsum(ftarray):
    (ndt, nxuv, nir, nz,nm, ndip)=shape(ftarray)
    retarray=zeros((nir,nz,nm), dtype='float')
    inplist=[0,1]
    indxlist=ftindxlist(inplist)
    for i,j,k in iterproduct(range(nir), range(nz), range(nm)):
        retarray[i,j,k]=sum(abs(ftarray[indxlist[0], indxlist[1],i,j,k,:])**2)
    return retarray

    
def thetaphigrid(ntheta=100, nphi=100):
    thetaarray, phiarray=mgrid[0:pi:ntheta*1j, 0:2*pi:nphi*1j]
    return thetaarray, phiarray

#def sphericalharmonics(thetaarray,phiarray,lmax,mmin=-2, mmax=2):
##sets up 4d array containing spherical harmonics evaluated on theta and phi grids
#    mrange=range(mmin,mmax+1)
#    lrange=range(0,lmax+1)
#    ntheta,nphi=shape(thetaarray)
#    retarray=zeros((lmax+1,len(mrange), ntheta, nphi), dtype='complex')
#    for l in range(0,lmax+1):
#        for m in range(max(-l,mmin),min(l,mmax)+1):
#            il=lrange.index(l)
#            im=mrange.index(m)
#            retarray[il,im,:,:]=sph_harm(m,l,thetaarray, phiarray)
#    return retarray

def nphotonsolve(Npm,N0,m, polarization='z'):
#Np, N0, Nm are number of photons with m=1,0,-1, respectively
#Npm=number of photons from phase matching
#Npm=N0+Np+Nm for z polarization
#Npm=Np+Nm for x polarization
#here, Z=Np+Nm
    if(polarization=='z'):
        Z=Npm-N0
        Np=(Z+m)/2
        Nm=(Z-m)/2
    if(polarization=='x'):
        Zp=Npm
        Np=(Zp+m)/2
        Nm=(Zp-m)/2
    return [Np,N0,Nm]

def contraction_cartesiangrid(ftarray, ndt=0, xuvnum=0, polarization='z', **cartesiankwd):
    xarray,yarray,zarray=makecartesianarrays(**cartesiankwd)
    rarray,thetaarray,phiarray=xyztopolar(xarray,yarray,zarray)

    #lengths of various vectors will be needed in order to loop correctly
    nxuv=len(veclist[list(namelist).index("xuvphasearray")])
    nIR=len(veclist[list(namelist).index("IRphasearray")])
    njtheta=len(veclist[list(namelist).index("jthetaphasearray")])
    nphiz=len(veclist[list(namelist).index("jphizphasearray")])
    nt=len(veclist[list(namelist).index("tvec")])

#vectors containing the photon numbers
    nxuvvec=nvec(nxuv)
    nIRvec=nvec(nIR)
    njthetavec=nvec(njtheta)
    nphizvec=nvec(nphiz)
    nxuvvec=nvec(nxuv)


    ixuv=nxuvvec.index(xuvnum)

    retvals=zeros(shape(xarray), dtype='complex')
#for every value of Np, N0, Nm, evaluate the partial wave contribution & add to the sum
#partial wave contribution = angular part times radial part
#evaluate spherical harmonics & raise to appropriate powers for angular part
#interpolate to get radial part

#precompute powers of the spherical harmonics evaluated on interpolation points
    matp=sph_harm(1,1,thetaarray,phiarray)
    mat0=sph_harm(0,1,thetaarray,phiarray)
    matm=sph_harm(-1,1,thetaarray,phiarray)
    tmpmat=ones(shape(matp),dtype='complex')
    matppowers=[tmpmat,matp]
    mat0powers=[tmpmat,mat0]
    matmpowers=[tmpmat,matm]
    for i in range(2,nIR+1):
        matppowers.append(matppowers[-1]*matp)
        mat0powers.append(mat0powers[-1]*mat0)
        matmpowers.append(matmpowers[-1]*matm)

    for i,j,k in iterproduct(range(nIR), range(njtheta), range(nphiz)):
        Ntot=nIRvec[i]
        Z=njthetavec[j]
        m=nphizvec[k]
        [Np,N0,Nm]=nphotonsolve(Ntot,Z,m,polarization=polarization)
        if((Np+N0+Nm)>=0):
            if(Np>=0):
                tmpmatp=matppowers[Np]
            else:
                tmpmatp=conjugate(matppowers[abs(Np)])
            if(N0>=0):
                tmpmat0=mat0powers[N0]
            else:
                tmpmat0=conjugate(mat0powers[abs(N0)])
            if(Nm>=0):
                tmpmatm=matppowers[Nm]
            else:
                tmpmatm=conjugate(matppowers[abs(Nm)])

            anglemat=tmpmatp*tmpmat0*tmpmatm
            radmat=onedinterp(rarray,freqvec,ftarray[ndt,ixuv,i,j,k,:])
            retvals+=anglemat*radmat
    return xarray, yarray, zarray, retvals
            


def sphericalcontraction(ftarray, ndt=0, xuvnum=0, ntheta=100, nphi=100, polarization='z'):
    thetaarray,phiarray=thetaphigrid(ntheta,nphi)
    nxuv=len(veclist[list(namelist).index("xuvphasearray")])
    nIR=len(veclist[list(namelist).index("IRphasearray")])
    njtheta=len(veclist[list(namelist).index("jthetaphasearray")])
    nphiz=len(veclist[list(namelist).index("jphizphasearray")])
    nt=len(veclist[list(namelist).index("tvec")])

    nxuvvec=nvec(nxuv)
    nIRvec=nvec(nIR)
    njthetavec=nvec(njtheta)
    nphizvec=nvec(nphiz)

    ixuv=nxuvvec.index(xuvnum)
    
    contractedarray=zeros((ntheta, nphi, nt/2), dtype='complex')
#do contraction
    matp=sph_harm(1,1,thetaarray,phiarray)
    mat0=sph_harm(0,1,thetaarray,phiarray)
    matm=sph_harm(-1,1,thetaarray, phiarray)

    for i,j,k in iterproduct(range(nIR), range(njtheta), range(nphiz)):
        Ntot=nIRvec[i]
        Z=njthetavec[j]
        m=nphizvec[k]
        [Np,N0,Nm]=nphotonsolve(Ntot,Z,m,polarization=polarization)
        if((Np+N0+Nm)>=0):
            anglemat=multiphotonmat(Np,N0,Nm, matp, mat0, matm)
            contractedarray+=arrayouter(anglemat,ftarray[ndt, ixuv,i,j,k,(nt/2):])
    return contractedarray



def multiphotonmat(Np,N0,Nm, matp, mat0, matm):
    if(Np<0):
        tmpmatp=-conjugate(matp)
    else:
        tmpmatp=matp
    if(N0<0):
        tmpmat0=conjugate(mat0)
    else:
        tmpmat0=mat0
    if(Nm<0):
        tmpmatm=-conjugate(matm)
    else:
        tmpmatm=matm
    return pow(tmpmatp,abs(Np))*pow(tmpmat0,abs(N0))*pow(tmpmatm,abs(Nm))


def threeYLMproduct(lmlist,ylmarray, lrange, mrange):
    indxlist=[[lrange.index(lm[0]), mrange.index(lm[1])] for lm in lmlist]
    (nl,nm,ntheta,nphi)=shape(ylmarray)
    retarray=ylmarray[indxlist[0][0],indxlist[0][1],:,:]
    for i in range(1,len(indxlist)):
        retarray*=ylmarray[indxlist[i][0],indxlist[i][1],:,:]
    return retarray
    
################################################################################
#plot contracted figure
#def plotcontracted(ftarray,ndt=0, xuvnum=1, ntheta=10, nphi=10, polarization='z'):
#    contractedarray=sphericalcontraction(ftarray=ftarray, ndt=ndt,
#                                         xuvnum=xuvnum, ntheta=ntheta,
#                                         nphi=nphi, polarization=polarization)
#    print("shape contractedarray\t"+str(shape(contractedarray)))
#    x,y,z=makecartesianarrays(ntheta=ntheta, nphi=nphi)
#    r=sqrt(x**2+y**2+z**2)
#    #contractedarray=exp(-(r-1)**2)
#    s=abs(contractedarray)
##    return mlab.pipeline.volume(mlab.pipeline.scalar_field(x,y,z,abs(contractedarray)))
#    src=mlab.pipeline.scalar_scatter(x,y,z,s)
#    gs=mlab.pipeline.gaussian_splatter(src)
#    gs.filter.radius=0.05
#    o=mlab.pipeline.outline(gs)
#    cp=mlab.pipeline.scalar_cut_plane(gs)
#    iso=mlab.pipeline.iso_surface(gs)
#    #return mlab.points3d(x,y,z,abs(contractedarray))
    
def plotcontracted(ftarray, ndt=0, xuvnum=1, polarization='z', **kwargs):
    scalarfield=contractedscalarfield(ftarray=ftarray, ndt=ndt, xuvnum=xuvnum,
                                    polarization=polarization, **kwargs)
    return mlab.pipeline.image_plane_widget(scalarfield,
                                            plane_orientation='x_axes',
                                            slice_index=10)

def contractedscalarfield(ftarray, ndt=0, xuvnum=1, polarization='z', **kwargs):
    x, y, z, vals=contraction_cartesiangrid(ftarray=ftarray,
                                                               ndt=ndt,
                                                               xuvnum=xuvnum,
                                                               polarization=polarization,
                                                               kwargs=kwargs)

    s=abs(vals)
    s/=s.max()
    scalarfield=makescalarfield(x,y,z,s)#mlab.pipeline.scalar_field(x,y,z,s)
    return x,y,z,s,scalarfield

def makescalarfield(x,y,z,s):
    return mlab.pipeline.scalar_field(x,y,z,s)

def plotcontours(ftarray, ndt=0, xuvnum=1, polarization='z', **kwargs):
#simple contourplot of scalar field created in the script
    x,y,z,s,scalarfield=contractedscalarfield(ftarray=ftarray, ndt=ndt, xuvnum=xuvnum,
                                    polarization=polarization, **kwargs)
    dx=.1
    ncontours=int(ceil(1./dx))
    return mlab.pipeline.iso_surface(scalarfield, contours=ncontours, opacity=.1)

def plot_contours_contourlist(x,y,z,s, contourlist=[.9], logplot=False,
                              logrange=log(1e3), **kwargs):
#plot specified contours of set s, with opacity ranging from 0 for smallest to
#1 for largest values
    if(logplot):
        maxval=log(abs(s).max())
        minval=maxval-logrange
        logs=log(abs(s))
        logs=where(logs>minval,logs-minval,0)
        logs/=logs.max()
        src=mlab.pipeline.scalar_field(x,y,z,logs)
    else:
        maxval=abs(s).max()
        src=mlab.pipeline.scalar_field(x,y,z,abs(s)/maxval)
    #add phase of inparray as an additional array (tricky!)
    #see example at
    #http://docs.enthought.com/mayavi/mayavi/auto/example_atomic_orbital.html#example-atomic-orbital
    src.image_data.point_data.add_array(angle(s).T.ravel())
    #give it the name 'angle'
    src.image_data.point_data.get_array(1).name='angle'
    #make sure the dataset is up to date
    src.image_data.point_data.update()
    abscontlist=[]
    for val in contourlist:
        mlab.pipeline.set_active_attribute(src,point_scalars='scalar')
        tmpcontour=mlab.pipeline.iso_surface(src, contours=[val], opacity=val)#.2+val*.6)
        #mlab.pipeline.set_active_attribute(src, point_scalars='angle')
        #surf2=mlab.pipeline.iso_surface(tmpcontour, colormap='hsv', vmin=-pi, vmax=pi)
#        surf2=mlab.pipeline.set_active_attribute(tmpcontour, point_scalars='angle')
#        mlab.pipeline.surface(surf2, colormap='hsv', vmin=-pi, vmax=pi)
        abscontlist.append(tmpcontour)
#    surf=mlab.pipeline.iso_surface(src,contours=contourlist, opacity=.1)
#    mlab.pipeline.set_active_attribute(surf, point_scalars='angle')
#    mlab.pipeline.iso_surface(surf,colormap='hsv', vmin=-pi, vmax=pi)


    mlab.colorbar(title='norm', orientation='vertical', nb_labels=3)
    #mlab.axes()#cube-shaped axes surrounding data
    mlab.orientation_axes()
    #mlab.show()
    return src

def plot_cutplane(x,y,z,s, logplot=False, logrange=log(1e3), **kwargs):
#plot specified contours of set s, with opacity ranging from 0 for smallest to
#1 for largest values
    if(logplot):
        maxval=log(abs(s).max())
        minval=maxval-logrange
        logs=log(abs(s))
        logs=where(logs>minval,logs-minval,0)
        logs/=logs.max()
        src=mlab.pipeline.scalar_field(x,y,z,logs)
    else:
        maxval=abs(s).max()
        src=mlab.pipeline.scalar_field(x,y,z,abs(s)/maxval)
    mlab.pipeline.image_plane_widget(src, **kwargs)
    mlab.colorbar(title='norm', orientation='vertical', nb_labels=3)
    mlab.outline()
    return src



#def plot_contours_colorphase(x,y,z,s, contourlist=[.9], **kwargs):
#    #plot contours using complex phase to give color information
#    src=mlab.pipeline.scalar_field(x,y,z,abs(s))
#    #add phase of inparray as an additional array (tricky!)
#    #see example at
#    #http://docs.enthought.com/mayavi/mayavi/auto/example_atomic_orbital.html#example-atomic-orbital
#    src.image_data.point_data.add_array(angle(s).T.ravel())
#    #give it the name 'angle'
#    src.image_data.point_data.get_array(1).name='angle'
#    #make sure the dataset is up to date
#    src.image_data.point_data.update()
#
#    #select the 'scalar' attribute
#    src2=mlab.pipeline.set_active_attribute(src,point_scalars='scalar')
#    #cut isosurfaces of the norm
#    contour=mlab.pipeline.contour(src2)
#    #now select 'angle' attibute (ie, phase)
#    contour2=mlab.pipeline.set_active_attribute(contour, point_scalars='angle')
#    #display the surface, with the colormap as the current attribute
#    mlab.pipeline.surface(contour2,colormap='hsv',vmin=-pi, vmax=pi)
#    mlab.colorbar(title='phase', orientation='vertical', nb_labels=3)
#    mlab.show()
#

def plot_volume(x, y, z, s, logplot=False, **kwargs):
    if(logplot):
        maxval=log(abs(s).max())
        minval=maxval-logrange
        logs=log(abs(s))
        logs=where(logs>minval,logs-minval,0)
        logs/=logs.max()
        src=mlab.pipeline.scalar_field(x,y,z,logs)
    else:
        maxval=abs(s).max()
        src=mlab.pipeline.scalar_field(x,y,z,abs(s)/maxval)
    
    vol=mlab.pipeline.volume(src)
    ctf=ColorTransferFunction()
    ctf.range=[0,1]
    otf=PiecewiseFunction()
    otf.add_point((0,0))
    otf.add_point((1,1))
    otf.range=[0,1]
    vol.otf=otf
    vol.volume_property.set_scalar_opacity(otf)
    return vol

def animate_rotateazimuth(nframes=36, framerate=10, figstr="fig", movietitle="animation.mov",
                          deletetmpfiles=False):
#creates a movie by rotating a figure about the z axis
    fig=mlab.gcf()
    tmpfiles=[]
    for i in range(nframes):
        fig.scene.camera.azimuth(260/nframes)
        fig.scene.render()
        fignamestr=figstr+"%03d" % i+".png"
        mlab.savefig(fignamestr)
        tmpfiles.append(fignamestr)
    commandstr="ffmpeg -framerate "+str(framerate)+" -i "+figstr+"%03d.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 23 -pix_fmt yuv420p -r 30 "+movietitle
    subprocess.call(commandstr, shell=True)
    if(deletetmpfiles):
        for filename in tmpfiles:
            os.remove(filename)


    




################################################################################
#load pickled arrays
def loaddata():
    freqvec=load("freqvec.pkl")
    zdiparray=load("zdiparray.pkl")

    xdiparray=load("xdiparray.pkl")
    pulsezarray=load("pulsezarray.pkl")
    pulsexarray=load("pulsexarray.pkl")

    ftzdiparray=load("ftzdiparray.pkl")
    ftxdiparray=load("ftxdiparray.pkl")
    ftpulsezarray=load("ftpulsezarray.pkl")
    ftpulsexarray=load("ftpulsexarray.pkl")

    namelist=load("namelist.pkl")
    veclist=load("veclist.pkl")
    ftaxislist=load("ftaxislist.pkl")#list of axes over which ft has been performed
    ftveclist=load("ftveclist.pkl")#index vectors for transformed array
    return [freqvec, zdiparray, xdiparray, pulsezarray, ftzdiparray, ftxdiparray,
            ftpulsezarray, ftpulsexarray, namelist, veclist, ftaxislist,
            ftveclist]

def contract(ngridpts, boxsize=1.4, polarization='z'):
    if(polarization=='z'):
        x,y,z,s=contraction_cartesiangrid(ftzdiparray,ndt=0,xuvnum=1,
                                          polarization='z', xmin=-boxsize,
                                          xmax=boxsize, nx=ngridpts, ymin=-boxsize,
                                          ymax=boxsize, ny=ngridpts, zmin=-boxsize,
                                          zmax=boxsize, nz=ngridpts)
    if(polarization=='x'):
         x,y,z,s=contraction_cartesiangrid(ftxdiparray,ndt=0,xuvnum=1,
                                          polarization='x', xmin=-boxsize,
                                          xmax=boxsize, nx=ngridpts, ymin=-boxsize,
                                          ymax=boxsize, ny=ngridpts, zmin=-boxsize,
                                          zmax=boxsize, nz=ngridpts)
    if(polarization=='p'):
        x, y, z,s0=contract(ngridpts=ngridpts, boxsize=boxsize, polarization='z')
        x1, y1, z1,s1=contract(ngridpts=ngridpts, boxsize=boxsize, polarization='x')
        s=s0+1j*s1
    if(polarization=='m'):
        x, y, z,s0=contract(ngridpts=ngridpts, boxsize=boxsize, polarization='z')
        x1, y1, z1,s1=contract(ngridpts=ngridpts, boxsize=boxsize, polarization='x')
        s=s0-1j*s1
                                          
    return x,y,z,s


################################################################################
##main program
[freqvec, zdiparray, xdiparray, pulsezarray, ftzdiparray, ftxdiparray, ftpulsezarray,
 ftpulsexarray, namelist, veclist, ftaxislist, ftveclist] = loaddata()

#plot_contours_colorphase(x,y,z,s)
x,y,z,s=contract(10, polarization='x')

#plot contours
#plot_contours_contourlist(x,y,z,s,arange(.05,1.,.05),logplot=False)
