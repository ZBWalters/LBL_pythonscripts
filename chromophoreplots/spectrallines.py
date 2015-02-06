from numpy import *
from scipy.integrate import quad,quadrature
from scipy import pi

#####constants
Hcm=2.1947e5
kboltz=3.16697647924e-6
aut=2.418884326505e-17

temp=300.#300.
kbT=temp*kboltz
Gamma=kbT

####helper functions
def C(w):
    return exp(-w**2.)

def Cmode1(w,lmbda,gamma):
    if(w == 0.):
        return 0.
    else:
        return 2*lmbda*(w*gamma)/(w**2+gamma**2)


def Cmode2(w,wj,gammaj,Sj):
    lmbda=Sj*wj
    return 2*lmbda*(wj**2)*(w*gammaj)/((wj**2.-w**2)**2.+(w**2)*(gammaj**2.))

def coth(z):
    coth=(exp(2*z)+1.)/(exp(2*z)-1.)
    return coth

def rintegrand(w,t,kbT):
    return (1.)/(2*pi*(w**2))*(coth(w/kbT)*(cos(w*t)-1))

def iintegrand(w,t,kbT):
    return -1./(2*pi*(w**2))*(sin(w*t)-w*t)

def grintegrand1(w,t,lmbda,gamma,kbT):
    eps=1e-3*gamma
    if(abs(w) < eps ):
        return grintegrand1(eps,t,lmbda,gamma,kbT)
    else:
        return rintegrand(w,t,kbT)*Cmode1(w,lmbda,gamma)

def grintegrand2(w,t,wj,gammaj,Sj,kbT):
    eps=1e-3*gammaj
    if(abs(w) < eps ):
        return grintegrand1(eps,t,lmbda,gamma,kbT)
    else:
        return rintegrand(w,t,kbT)*Cmode2(w,wj,gammaj,Sj)

def giintegrand1(w,t,lmbda,gamma,kbT):
    return iintegrand(w,t,kbT)*Cmode1(w,lmbda,gamma)

def giintegrand2(w,t,wj,gammaj,Sj,kbT):
    return iintegrand(w,t,kbT)*Cmode2(w,wj,gammaj,Sj)

def gr(t,lambdalist,gammalist,wjlist,Sjlist,gammaj,kbT):
    g=0.
    #overdamped modes
    for i in range(len(lambdalist)):
        lmbda=lambdalist[i]
        gamma=gammalist[i]
        lim=10.*gamma
        (gtmp1,gtmperr1)=quad(grintegrand1,-lim,0.,args=(t,lmbda,gammaj,kbT))
        (gtmp2,gtmperr2)=quad(grintegrand1,0,lim,args=(t,lmbda,gammaj,kbT))
        g=g+gtmp1+gtmp2
        
    for i in range(len(wjlist)):
        wj=wjlist[i]
        Sj=Sjlist[i]
        lim=max(gammaj,wj)*10.
        (gtmp1,gtmperr1)=quad(grintegrand2,-lim,0,args=(t,wj,gammaj,Sj,kbT))
        (gtmp2,gtmperr2)=quad(grintegrand2,0,lim,args=(t,wj,gammaj,Sj,kbT))
        g=g+gtmp1+gtmp2
    return g

def gi(t,lambdalist,gammalist,wjlist,Sjlist,gammaj,kbT):
    g=0.
    #overdamped modes
    for i in range(len(lambdalist)):
        lmbda=lambdalist[i]
        gamma=gammalist[i]
        lim=10.*gamma
        (gtmp1,gtmperr1)=quadrature(giintegrand1,-lim,0.,args=(t,lmbda,gamma,kbT))
        (gtmp2,gtmperr2)=quadrature(giintegrand1,0,lim,args=(t,lmbda,gamma,kbT))
        g=g+gtmp1+gtmp2
        
    for i in range(len(wjlist)):
        wj=wjlist[i]
        Sj=Sjlist[i]
        lim=max(gammaj,wj)*10.
        (gtmp1,gtmperr1)=quadrature(giintegrand2,w-lim,w,args=(t,wj,gammaj,Sj,kbT))
        (gtmp2,gtmperr2)=quadrature(giintegrand2,w,w+lim,args=(t,wj,gammaj,Sj,kbT))
        g=g+gtmp1+gtmp2
    return g

#PE545 overdamped modes
lambdalist=[40,70]
gammalist=[30,90]


#PE545 resonant modes
wjlist=[207,244,312,372,438,514,718,813,938,1111,1450,1520,1790,2090]
Sjlist=[.0013,.0072,.0450,.0578,.0450,.0924,.0761,.0578,.0313,.0578,.1013,.0265,.0072,.0113]

gammaj=20
#convert to atomic units
wjlist=list(array(wjlist)/Hcm)
lambdalist=list(array(lambdalist)/Hcm)
gammalist=list(array(gammalist)/Hcm)

lambdasum=0.
for i in range(len(lambdalist)):
    lambdasum=lambdasum+lambdalist[i]
for i in range(len(Sjlist)):
    lambdasum=lambdasum+Sjlist[i]*wjlist[i]

gammaj=gammaj/Hcm



#########Main program
tarray=range(11)
#grarray=gr(tarray,lambdalist,gammalist,wjlist,Sjlist,gammaj,kbT)
grarray=list(array(tarray)*0.)
for i in range(len(tarray)):
    grarray[i]=gr(tarray[i],lambdalist,gammalist,wjlist,Sjlist,gammaj,kbT)
    

##test integrand
#warray=arange(-gammalist[0],gammalist[0],gammalist[0]/10000)
#gintarray=zeros(shape(warray))
#for i in range(len(warray)):
#    gintarray[i]=grintegrand1(warray[i],100.,lambdalist[0],gammalist[0],kbT)
#plot(warray,gintarray)
#show()
