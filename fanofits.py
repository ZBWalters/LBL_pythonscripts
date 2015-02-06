from numpy import *
from scipy import *
from scipy.optimize import curve_fit
from pylab import plot,figure,savefig


def datfromfile(filename):
    loadarray=loadtxt(filename)
    xdat=loadarray[:,0]
    ydat=loadarray[:,1]
    zdat=loadarray[:,2]+1j*loadarray[:,3]
    return [xdat,ydat,zdat]

def uniquevals(inplist):
    #retlist=[]
    #for elt in inplist:
    #    if (not (elt in retlist)):
    #        retlist.append(elt)
    #retlist.sort()
    #return retlist
    return sort(list(set(inplist)))

def dattoarrays(xdat,ydat,zdat):
    n=len(uniquevals(xdat))
    m=len(uniquevals(ydat))
    xarray=reshape(xdat,(n,m))
    yarray=reshape(ydat,(n,m))
    yvec=yarray[0,:]
    zarray=reshape(zdat,(n,m))
    return [xarray,yarray,zarray]
########################################
def xyzmax(xarray,yarray,zarray):
    #return coordinates and value of highest peak
    #maxindx=abs(zarray).argmax()
    #zval=list(zarray)[maxindx]
    #xval=list(xarray)[maxindx]
    #yval=list(yarray)[maxindx]
    #xvec=sort(list(set(xarray)))
    #yvec=sort(list(set(yarray)))
    #xindx=list(xvec).index(xval)
    #yindx=list(yvec).index(yval)
    xindx,yindx=unravel_index(abs(zarray).argmax(),zarray.shape)
    print("xyzmax indices\t"+str(xindx)+"\t"+str(yindx))
    zval=zarray[xindx,yindx]
    return [xindx,yindx,zval]


#def fano(x,amp,x0,gam,q):
#    num=pow(q*gam/2+(x-x0),2)
#    den=pow(gam/2,2)+pow(x-x0,2)
#    return amp/pow(q,2)*num/den

#def fano(x,amp,x0,gam,atanq):
#    q=tan(atanq)
#    num=pow(q*gam/2+(x-x0),2)
#    den=pow(gam/2,2)+pow(x-x0,2)
#    return amp/pow(q,2)*num/den


#def fano(x,amp,x0,gam,q):
#    eps=(x-x0)/(gam/2)
#    return amp*(pow(q+eps,2)/pow(q,2))/(1+pow(eps,2))

#def fano(x,amp,x0,gam,gambyq):
#    num=pow(gam,2)/(2*gambyq)+x-x0
#    den=pow(gam/gambyq,2)*(pow(gam/2,2)+pow(x-x0,2))
#    return amp*pow(num,2)/den

def fanop(x,qsqamp,x0,gam,qinv):
    return qsqamp*pow(gam/2+(x-x0)*qinv,2)/(pow(gam/2,2)+pow(x-x0,2))- \
        qsqamp*pow(qinv,2)

def fanoparamconvert(qsqamp,x0,gam,qinv):
    #convert from variables used in fanop to variables used in fano
    amp=qsqamp*pow(qinv,2)
    q=1./qinv
    back=-qsqamp*pow(qinv,2)
    return [amp,x0,gam,q,back]

def fano(x,amp,x0,gam,q,back):
    num=pow(q*gam/2+(x-x0),2)
    den=pow(gam/2,2)+pow(x-x0,2)
    return amp*num/den+back

def absfano(x,amp,x0,gam,q,back):
    return abs(fano(x,amp,x0,gam,q,back))

def lorentz(x,amp,x0,gam):
    return amp*pow(.5*gam,2)/(pow(x-x0,2)+pow(.5*gam,2))


def fano2(x,arglist):
    return fano(x,*arglist)

def fano2p(x,arglist):
    return fanop(x,*arglist)

#def fanofit(xarray,yarray):
#    index_max=abs(yarray).argmax()
#    yval=yarray[index_max]
#    xval=xarray[index_max]
#    gammatry=1.
#    qtry=100.
#    [amptry,x0try,gamtry],cov=curve_fit(lorentz,xarray,abs(yarray),p0=[abs(yval),xval,1.],maxfev=100000)
#    #ptry=[yval/pow(qtry,2),xval+qtry/2,gammatry,qtry,0]
#    fixedgamma = lambda x,amp,x0,q,back: fano(x,amp,x0,gamtry,q,back)
#    ptry=[amptry,x0try,1.,0.]
#    [amptry,x0try,qtry,backtry], cov=curve_fit(fixedgamma,xarray,yarray,p0=ptry,maxfev=10000)
#    arglist=[amptry,x0try,gamtry,qtry,backtry]
#
#    ptry=[amptry,x0try,gamtry,qtry,backtry]
#    arglist,cov=curve_fit(fano,xarray,yarray,p0=ptry,maxfev=100000)
#    #arglist=[amptry,x0try,gamtry,100,0]
#    return arglist

def fanofit(xarray,yarray):
    arglistp=fanopfit(xarray,yarray)
    arglist=fanoparamconvert(*arglistp)
    return arglist

def fanopfit(xarray,yarray):
    #plot(xarray,yarray)
    #print("xarray fanopfit\t"+str(xarray))
    #print("yarray fanopfit\t"+str(yarray))
    index_max=abs(yarray).argmax()
    yval=yarray[index_max]
    xval=xarray[index_max]
    qsqamptry=abs(yval)
    x0try=xval
    gamtry=.01
    qinvtry=0.01
    backtry=0.
    ptry=[qsqamptry,x0try,gamtry,qinvtry]#,backtry]
    print("ptry\t"+str(ptry))
    arglist,cov=curve_fit(fanop,xarray,yarray,p0=ptry,maxfev=100000)
    #[qsqamp,x0,gam,qinv,back], cov=curve_fit(fanop,xarray,yarray,p0=ptry,maxfev=10000)
    #amp=qsqamptry*pow(qinv,2)
    #q=1/qinv
    #arglist=[amp,x0,gam,q,back]
    return arglist

def twoDfano(inpdat,amp2D,xamp,x0,xgam,xq,xback,yamp,y0,ygam,yq,yback):
    x,y=inpdat
    return amp2D*fano(x,xamp,x0,xgam,xq,xback)*fano(y,yamp,y0,ygam,yq,yback)

def twoDfanofit(xarray,yarray,zarray):
    [xindx,yindx,zval]=xyzmax(xarray,yarray,zarray)
    [xamp,x0,xgam,xq,xback]=fanofit(xarray[:,yindx],zarray[:,yindx])
    #print("fit x")
    #fig1=figure()
    #plot(xarray[:,yindx],zarray[:,yindx])
    #plot(xarray[:,yindx],fano(xarray[:,yindx],xamp,x0,xgam,xq,xback))

    [yamp,y0,ygam,yq,yback]=fanofit(yarray[xindx,:],zarray[xindx,:])
    #fig2=figure()
    #plot(yarray[xindx,:],zarray[xindx,:])
    #print("fit y")
    #plot(yarray[xindx,:],fano(yarray[xindx,:],yamp,y0,ygam,yq,yback))
    ampfun = lambda inpdat, amp2D:twoDfano(inpdat,amp2D,xamp,x0,xgam,xq,xback,yamp,y0,ygam,yq,yback)
    [amp2D],cov=curve_fit(ampfun,[xarray.flat,yarray.flat],zarray.flat,p0=[1./sqrt(abs(xamp*yamp))*sign(xamp)])
    #amp2D=1.
    retarglist=[amp2D,xamp,x0,xgam,xq,xback,yamp,y0,ygam,yq,yback]
    
    #iteratively refine fit in x and y
    #for i in range(3):
    #    retarglist=refinexfanofit(xarray,yarray,zarray,retarglist)
    #    retarglist=refineyfanofit(xarray,yarray,zarray,retarglist)

    #ampfixed=lambda inpdat,amp2D,x0,xgam,xq,xback,y0,ygam,yq,yback:\
    #          twoDfano(inpdat,amp2D,xamp,x0,xgam,xq,xback,yamp,y0,ygam,yq,yback)
    #ptry=[amp2D,x0,xgam,xq,xback,y0,ygam,yq,yback]
    #[amp2D,x0,xgam,xq,xback,y0,ygam,yq,yback],cov= curve_fit(ampfixed,[xarray.flat,yarray.flat],zarray.flat,p0=ptry,maxfev=10000)
    #retarglist=[amp2D,xamp,x0,xgam,xq,xback,yamp,y0,ygam,yq,yback]
    
    return retarglist

def refinexfanofit(xarray,yarray,zarray,arglist):
    [tryamp2D,tryxamp,tryx0,tryxgam,tryxq,tryxback,tryyamp,\
     tryy0,tryygam,tryyq,tryyback]=arglist
    fixedy=lambda inpdat,xamp,x0,xgam,xq,xback:\
            twoDfano(inpdat,tryamp2D,xamp,x0,xgam,xq,xback,\
                     tryyamp,tryy0,tryygam,tryyq,tryyback)
    pstart=[tryxamp,tryx0,tryxgam,tryxq,tryxback]
    [tryxamp,tryx0,tryxgam,tryxq,tryxback],cov=curve_fit(fixedy,[xarray.flat,yarray.flat],zarray.flat,p0=pstart,maxfev=100000)
    return [tryamp2D,tryxamp,tryx0,tryxgam,tryxq,tryxback,tryyamp,\
     tryy0,tryygam,tryyq,tryyback]

def refineyfanofit(xarray,yarray,zarray,arglist):
    [tryamp2D,tryxamp,tryx0,tryxgam,tryxq,tryxback,tryyamp,\
     tryy0,tryygam,tryyq,tryyback]=arglist
    fixedx=lambda inpdat,yamp,y0,ygam,yq,yback:\
            twoDfano(inpdat,tryamp2D,tryxamp,tryx0,tryxgam,tryxq,tryxback,\
                     yamp,y0,ygam,yq,yback)
    pstart=[tryyamp,tryy0,tryygam,tryyq,tryyback]
    [tryyamp,tryy0,tryygam,tryyq,tryyback], cov=curve_fit(fixedx,[xarray.flat,yarray.flat],zarray.flat,p0=pstart,maxfev=10000)
    return [tryamp2D,tryxamp,tryx0,tryxgam,tryxq,tryxback,tryyamp,\
     tryy0,tryygam,tryyq,tryyback]

    

def printtwoDfanofitlist(fitlist):
    Hrt=27.21
    for i in range(len(fitlist)):
        [amp2D,xamp,x0,xgam,xq,xback,yamp,y0,ygam,yq,yback]=fitlist[i]
        print("fit "+str(i)+"\tamp "+str(twoDfano([x0,y0],*fitlist[i]))+"\tx0, y0\t"+str(x0*Hrt)+", "+str(y0*Hrt)+"\t xgam, ygam\t"+str(xgam*Hrt)+", "+str(ygam*Hrt)+"\txq, yq\t"+str(xq)+", "+str(yq))
        #print("fit "+str(i)+"\tx0, y0\t"+str(x0*Hrt)+", "+str(y0*Hrt)+"\t xgam, ygam\t"+str(xgam*Hrt)+", "+str(ygam*Hrt)+"\txq, yq\t"+str(xq)+", "+str(yq))

def twoaxisplot(xarray,yarray,zarray,xindx,yindx,arglist,indx):
    xvec=xarray[:,yindx]
    zxvec=zarray[:,yindx]

    yvec=yarray[xindx,:]
    zyvec=zarray[xindx,:]
    
    fig1=figure()
    plot(xvec,zxvec)
    plot(xvec,twoDfano([xvec,yarray[:,yindx]],*arglist))
    savefig("peak_"+str(indx)+"_x.png")
    fig2=figure()
    plot(yvec,zyvec)
    plot(yvec,twoDfano([xarray[xindx,:],yvec],*arglist))
    savefig("peak_"+str(indx)+"_y.png")

def iterativetwoDfanofit(xarray,yarray,zarray,nmax):
    fitlist=[]
    tmpzarray=zarray*1.
    for i in range(nmax):
        arglist=twoDfanofit(xarray,yarray,tmpzarray)
        
        #plot fits
        [xindx,yindx,zval]=xyzmax(xarray,yarray,tmpzarray)
        twoaxisplot(xarray,yarray,tmpzarray,xindx,yindx,arglist,i)
        
        fitlist.append(arglist)
        tmpzarray=tmpzarray-twoDfano([xarray,yarray],*arglist)
        #twoaxisplot(xarray,yarray,tmpzarray,xindx,yindx,arglist)
    return fitlist,tmpzarray

def iterativetwoDfanofitfile(filename,nmax):
    [xdat,ydat,zdat]=datfromfile(filename)
    [xarray,yarray,zarray]=dattoarrays(xdat,ydat,zdat)
    fitlist,tmpzarray=iterativetwoDfanofit(xarray,yarray,real(zarray),nmax)
    return fitlist, xarray,yarray,tmpzarray

#def fanofit(xarray,yarray):
#    tmparglist=lorentzianfit(xarray,yarray)
#    [amp,x0,gam]=tmparglist
#    ptry=[amp,x0,gam,gam]
#    arglist,cov=curve_fit(fano,xarray,yarray,p0=ptry,maxfev=100000)
#    return arglist

def iterativefanofit(xarray,yarray,maxn):
    fitlist=[]
    tmpyarray=yarray*1.
    for i in range(maxn):
        arglist=fanofit(xarray,tmpyarray)
        fitlist.append(arglist)
        tmpyarray=tmpyarray-fano2(xarray,arglist)
    return fitlist,tmpyarray
