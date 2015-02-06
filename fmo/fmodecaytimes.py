from numpy import *
from scipy import *

#constants
wbyev=1/8065.54445
wbyh=wbyev/27.21

evkb=8.6173423e-5
hkb=evkb/27.21
wkb=evkb/wbyev

aut=24.1e-18

temp=300.

fmo=matrix([[280,-106,8,-5,6,-8,-4],[-106,420,28,6,2,13,1],[8,28,0,-62,-1,-9,17],[-5,6,-62,175,-70,-19,-57],[6,2,-1,-70,320,40,-2],[-8,13,-9,-19,40,360,32],[-4,1,17,-57,-2,32,260]])


def expmat(mat,kb,temp):
    (n0,n1)=shape(mat)
    outmat=zeros((n0,n1))
    for i in range(n0):
        for j in range(n1):
            outmat[i,j]=mat[i,j]*exp(-abs(mat[i,i]-mat[j,j])/(kb*temp))
    return outmat

def decayconst(mat,kb,temp):
    (n0,n1)=shape(mat)
    outmat1=zeros((n0,n1))
    outmat2=zeros((n0,n1))
    for i in range(n0):
        for j in range(n1):
            outmat1[i,j]=(-kb*temp+((kb*temp)**2-16*mat[i,j]**2+0j)**.5)/2
            outmat2[i,j]=(-kb*temp-((kb*temp)**2-16*mat[i,j]**2+0j)**.5)/2
    return (outmat1,outmat2)

def decaytimes(hmat,aut):#matrix must have units of hartree
    (n0,n1)=shape(hmat)
    outmat=zeros((n0,n1))
    for i in range(n0):
        for j in range(n1):
            outmat[i,j]=aut/hmat[i,j]
    return outmat


fmoexp=expmat(fmo,wkb,temp)*wbyh
(fmodecay1,fmodecay2)=decayconst(fmoexp,hkb,temp)
fmodecaytimes=decaytimes(fmodecay1,aut)
