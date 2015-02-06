from numpy import *
from scipy import *
#from itertools import *
import itertools

Hcm=2.1947e5
kboltz=3.16697647924e-6
aut=2.418884326505e-17

######ps2 rc
#Vc=[[0,150,0,-42,-6,17],[150,0,-60,-36,21,-3],[-42,-60,0,7,47,-4],[-53,-36,7,0,-5,35],[-6,21,47,-5,0,3],[17,-3,-4,35,3,0]]
#trialE=[666,666,678,667,672,675]
#trialE=(45.565/array(trialE))*Hcm

####fmo
#Vc=[[0,-109,6.9,-7.4,8.4,-17.1,-12.3],[-109.6,0,38.5,10.2,0.8,14.8,5.4],[6.9,38.5,0,-66.8,-2.7,-12.0,7.5],[-7.4,10.2,-66.8,0,-88.4,-21.3,-79.1],[8.4,0.8,-2.7,-88.4,0,101.4,-1.6],[-17.2,14.8,-12.0,-21.3,101.4,0,49.6],[-12.3,5.4,7.5,-79.1,-1.6,49.6,0]]
#
#trialE=[12410,12530,12210,12320,12480,12630,12440]
#
#minresE=[ 12219.01737682,  11812.97866749,  11934.84677454,  12706.3980562,
#  13244.05186721,  12570.00907069,  12532.69818706]
#trialE=minresE
########PE545
Vc=[[0,1,-37,37,23,92,-16,12],[1,0,4,-11,33,-39,-46,3],[-37,4,0,45,3,2,-11,34],[37,-11,45,0,-7,-17,-3,6],[23,33,3,-7,0,18,7,6],[92,-39,2,-17,18,0,40,26],[-16,-46,-11,-3,7,40,0,7],[12,3,34,6,6,26,7,0]]

E=[18532,18008,17973,18040,18711,19574,19050,18960]
E68=[18532,18008,17973,18040,18781,18009,19030,19574]
E67=[18532,18008,17973,18040,18781,18809,19574,19030]
E65=[18532,18008,17973,18040,19574,18711,19030,18909]
E17=[19050,18008,17973,18040,18651,19574,18532,18050]
E47=[18641,18043,18030,19111,18499,19588,17909,18868]
E42=[18532,18048,18013,17940,18711,19574,18960,19050]

modellist=[E,E68,E67,E65,E17,E47,E42]

#orderings permuted to maximize number of pair decay times lt 10 ps
Eperm=[ 19574.,18711.,19050.,18960.,18040.,18532.,17973.,18008.]
E42perm=[18711.,19574.,17940.,18048.,18960.,18532.,19050.,18013.]#most pairs permutation of E42
E47perm=[ 18641.,18043.,19111.,17909.,18499.,18030.,18868.,19588.]

#least square adjustments to (unpermuted) site energies to equlibrate decay times 
Ep=[ 18502.56469304,18147.0304042,17859.94522473,18152.09984185,18693.6037071,19456.77079372,18953.96975555,19082.01557982]
E42p=[ 18555.09861812,18062.65135937,18164.09143091,17842.54357043,18755.56896158,19523.65782979,18936.68323286,18987.70499694]
E47p=[ 18673.58842121,18169.8956041,18009.40438836,19206.97006875,18497.67959716,19511.99269605,17830.49618091,18788.97304346]


Etmp=[ 18522.99771106,  17950.72328974,  17900.56378318,  19227.19065155,
  18658.09808922,  19236.22969004,  18506.75423333,  19050.        ]
trialE=E#E42p#47#E42#E47




Vc=array(Vc)/Hcm
trialE=array(trialE)/Hcm
for i in range(len(modellist)):
    modellist[i]=array(modellist[i])/Hcm

temp=300.
kbT=temp*kboltz
Gamma=kbT


###########define helper functions
def decayconstarrays(Veff,Gamma):
    (n1,n2)=shape(Veff)
    lambdaslow=zeros((n1,n2))*0j
    lambdafast=zeros((n1,n2))*0j
    for i in range(n1):
        for j in range(n2):
            rad=((1+0j)*(Gamma**2.)-16.*(Veff[i,j]**2.))**.5
            #print "rad",i,j,rad
            lambdaslow[i,j]=(-Gamma+rad)/2
            lambdafast[i,j]=(-Gamma-rad)/2
    return (lambdaslow,lambdafast)

def Veff(Vc,siteE,kbT):
    (n1,n2)=shape(Vc)
    Veff=zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            Veff[i,j]=Vc[i,j]*exp(-abs(siteE[i]-siteE[j])/(2.*kbT))
    return Veff

def matcutoff(mat,cutoff):
    retmat=mat*0.
    (n1,n2)=shape(Vc)
    for i in range(n1):
        for j in range(n2):
            if(abs(mat[i,j]) < cutoff):
                retmat[i,j]=mat[i,j]
            else:
                retmat[i,j]=0.
    return retmat

def constpairs(lambdamat,cutoff):
    (n1,n2)=shape(lambdamat)
    pairlist=[]
    for i in range(n1):
        for j in range(i,n2):
            if(abs(lambdamat[i][j]) < cutoff):
                pairlist.append((i,j))
    return pairlist

def equaldecaytimes_pairs(Vc,trialE,pairlist,kbT):
    (n1,n2)=shape(Vc)
    npair=len(pairlist)
    if(npair == 0):
        return (trialE,0)
    neq=n1+npair
    nunk=n1+1# num energies + rate of decay
    Lmat=zeros((neq,nunk))
    Rvec=zeros(neq)
    #first set up newE[i]=trialE[i]
    cnt=0
    for i in range(n1):
        Lmat[cnt,i]=1.
        Rvec[cnt]=trialE[i]
        cnt=cnt+1
    #next, set up E1-E2 + kbT*logz = kbT*log(4 V**2)
    for i in range(npair):
        (i1,i2)=pairlist[i]
        Lmat[cnt,i1]=1.
        Lmat[cnt,i2]=-1.
        if(trialE[i1]>trialE[i2]):
            Lmat[cnt,-1]=kbT
            Rvec[cnt]=kbT*log(4*Vc[i1,i2]**2.)
        else:
            Lmat[cnt,-1]=-kbT
            Rvec[cnt]=-kbT*log(4*Vc[i1,i2]**2.)
        cnt=cnt+1
    (soln,res,rank,s)=linalg.lstsq(Lmat,Rvec)
    newE=trialE*0.
    for i in range(n1):
        newE[i]=soln[i]
    logzeta=soln[-1]
    decayconst=exp(logzeta-log(kbT))
    return (newE,decayconst)

def defineddecaytimes_pairs(Vc,trialE,pairdecays,kbT):
    (n1,n2)=shape(Vc)
    npair=len(pairlist)
    if(npair == 0):
        return (trialE,0)
    neq=npair#n1+npair
    nunk=n1# num energies 
    Lmat=zeros((neq,nunk))
    Rvec=zeros(neq)
     #first set up newE[i]=trialE[i]
    cnt=0
    #for i in range(n1):
    #    Lmat[cnt,i]=1.
    #    Rvec[cnt]=trialE[i]
    #    cnt=cnt+1
    #next, set up E1-E2 = kbT*(log(4 V**2)-log(kbT)-log(lambda)0
    for i in range(npair):
        (i1,i2)=pairdecays[i][0]
        lmbda=pairdecays[i][1]
        Lmat[cnt,i1]=1.
        Lmat[cnt,i2]=-1.
        dE=kbT*(log(4*Vc[i1,i2]**2.)-log(kbT)-log(lmbda))
        if(trialE[i1]>trialE[i2]):
            Rvec[cnt]=dE
        else:
            Rvec[cnt]=-dE
        cnt=cnt+1
    #print "Rvec",Rvec
    (soln,res,rank,s)=linalg.lstsq(Lmat,Rvec)
    print "res",res*Hcm
    newE=soln
    return newE

def leastsquareoffset(newE,oldE):
    n1=len(newE)
    neq=n1
    nunk=1
    Lmat=zeros((neq,nunk))
    Rvec=zeros(neq)
    cnt=0
    for i in range(n1):
        Lmat[cnt,0]=1
        Rvec[cnt]=oldE[i]-newE[i]
    (soln,res,rank,s)=linalg.lstsq(Lmat,Rvec)
    #print "offset",soln
    return soln

def Edist(E1,E2):
    n1=len(E1)
    dist=0.
    for i in range(n1):
        dist=dist+abs(E1[i]-E2[i])
    return dist/n1

def Esqdist(E1,E2):
    n1=len(E1)
    dist=0.
    for i in range(n1):
        dist=dist+abs(E1[i]-E2[i])**2.
    return sqrt(dist/n1)

def permute_mostpairs(Vc,trialE,kbT,decaycutoff):
    (n1,n2)=shape(Vc)
    Eperms=list(itertools.permutations(trialE))
    maxnpairs=0
    for Eperm in Eperms:
        Veffmat=Veff(Vc,Eperm,kbT)
        (lslow,lfast)=decayconstarrays(Veffmat,kbT)
        decaymat=1./lslow
        npairs=len(constpairs(decaymat,decaycutoff))
        if(npairs > maxnpairs):
            maxnpairs=npairs
            maxEperm=Eperm
    return (maxEperm,maxnpairs)
        
def adjustenergies_leastsquares(Vc,trialE,kbT,tcutoff):
    Veffmat=Veff(Vc,trialE,kbT)
    (lslow,lfast)=decayconstarrays(Veffmat,kbT)
    decaymat=1./lslow
    pairlist=constpairs(decaymat,tcutoff)
    (newE,decayconst)=equaldecaytimes_pairs(Vc,trialE,pairlist,kbT)
    #print "newE",newE*Hcm
    #print "decaytime",1/decayconst*aut
    #print "E distance",Edist(newE,trialE)*Hcm,Esqdist(newE,trialE)*Hcm
    return (newE,decayconst)

#####main program
#trialEp=trialE
#for i in range(100):
#    (trialEp,decayconst)=adjustenergies_leastsquares(Vc,trialEp,kbT,10e-12/aut)
#print "new E",trialEp*Hcm 
#print "E distance",Edist(trialEp,trialE)*Hcm,Esqdist(trialEp,trialE)*Hcm
#Veffmat=Veff(Vc,trialEp,kbT)
#(lslow,lfast)=decayconstarrays(Veffmat,Gamma)

#for i in range(len(modellist)):
#    trialE=modellist[i]
#print "trialE",trialE
Veffmat=Veff(Vc,trialE,kbT)

(lslow,lfast)=decayconstarrays(Veffmat,Gamma)

psmat=1./lslow*aut/1e-12
psmat2=1./lfast*aut/1e-12

pairlist=constpairs(psmat,10.)
print "pairlist",len(pairlist), pairlist

psmat=matcutoff(psmat,100.)
print "cutoff psmat",psmat


##adjust energies to experimental decay times
pairlist=[]
pairlist.append([(1,2),aut/22e-12])

pairlist.append([(0,2),aut/4e-12])

pairlist.append([(0,5),aut/1e-12])
pairlist.append([(0,4),aut/1e-12])
pairlist.append([(0,6),aut/1e-12])
pairlist.append([(5,7),aut/1e-12])
pairlist.append([(5,3),aut/1e-12])

print "pairlist",pairlist
newE=defineddecaytimes_pairs(Vc,trialE,pairlist,kbT)
newE=newE+leastsquareoffset(newE,trialE)
print "trialE",trialE*Hcm
print "newE",newE*Hcm
#print "offset",leastsquareoffset(newE,trialE)*Hcm
newE=newE+leastsquareoffset(newE,trialE)
print "Edist",Edist(newE,trialE)*Hcm
