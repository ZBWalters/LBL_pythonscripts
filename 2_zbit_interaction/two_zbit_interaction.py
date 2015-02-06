from numpy import *
from scipy import *

#order of eigenstates is
#(alpha1,alpha2),(alpha1,beta2),(beta1,alpha2),(beta1,beta2)

def Hsetup(delta1,delta2):
    Hmat=zeros((4,4))*0j
    Hmat[0,1]=delta2
    Hmat[1,0]=delta2
    
    Hmat[0,2]=delta1
    Hmat[2,0]=delta1

    Hmat[1,3]=delta1
    Hmat[3,1]=delta1

    Hmat[3,2]=delta2
    Hmat[2,3]=delta2

    return Hmat

def twodindx(val1,val0):
    return val0+val1*2
    

def Gammasetup(Gamma,indx,val):
    Gmat=zeros((4,4))*0j
    if(indx==0):
        for val1 in range(2):
            indx=twodindx(val1,val)
            Gmat[indx,indx]=Gamma/2
    if(indx==1):
        for val0 in range(2):
            indx=twodindx(val,val0)
            Gmat[indx,indx]=Gamma/2
    return Gmat

def Gammalistsetup(Gamma):
    Gammalist=[]
    for indx in range(2):
        for val in range(2):
            Gammalist.append(Gammasetup(Gamma,indx,val))
    return Gammalist
        
def Comm(A,B):
    return dot(A,B)-dot(B,A)

def threedot(A,B,C):
    return dot(A,dot(B,c))

def Lindblad(pmat,Lk):
    Lkdag=conjugate(transpose(Lk))
    return -threedot(pmat,Lkdag,Lk)-threedot(Lkdag,Lk,pmat)+2*threedot(Lk,pmat,Lkdag)
    

def pmatdot(pmat,t,Hmat,Glist):
    pdot= 1j*Comm(pmat,Hmat)
    for Lk in Glist:
        pdot+=Lindblad(pmat,Lk)
    return pdot

def pmatsetup_small(r,theta):
    Vz=r*cos(theta)
    Vx=r*sin(theta)
    sigmax=[[0,1],[1,0]]
    sigmay=[[0,-1j],[1j,0]]
    sigmaz=[[1,0],[0,-1]]
    pmat=eye(2)+Vz*sigmaz+Vx*sigmax
    return pmat

def pmatsetup_large(r0,phi0,r1,phi1):
    p0=pmatsetup_small(r0,phi0)
    p1=pmatsetup_small(r1,phi1)
    pmat=outer(p0,p1)
    return pmat



##################
pmat=pmatsetup_large(1.,)
