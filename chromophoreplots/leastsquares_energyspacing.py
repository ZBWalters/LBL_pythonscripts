from numpy import *
from scipy import *
#from itertools import *
import itertools

Hcm=2.1947e5
kboltz=3.16697647924e-6


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

Eminres=[ 18378.36614781,  19221.36295264,  18068.9928968,   18315.48121632,
  19151.11937559,  18815.89408202,  18904.79652228,  17991.98680654]

Ep=[ 18632.85352921,  18606.,          17873.41844246,  17934.74878147,  18606.,
  19233.18989207,  19125.4275776,   18836.36177719]

E42p=[ 18485.63232422,  17715.39764404,  18050.2822876 ,  17949.81689453,
        18941.07543945,  19356.33239746,  18954.4708252 ,  19034.84313965]

trialE=E#E42#E47



Vc=array(Vc)/Hcm
trialE=array(trialE)/Hcm

def deltaE(i,j,Vc,trialE,kbT):
    #dE=2*kbT*log(kbT/(4*abs(Vc[i,j])))
    if(abs(4*Vc[i,j]) > kbT):
        #dE=abs(2*kbT*(log(kbT)-log(4*abs(Vc[i,j]))))
        dE=2*kbT*(log(4*abs(Vc[i,j]))-log(kbT))
    else:
        dE=1e6
    if(trialE[j]>trialE[i]):
        dE=-dE
        #dE=dE
    else:
        dE=dE
        #dE=-dE
    return dE*Hcm

def solveE(Vc,trialE,kbT,Hcm):
    (n1,n2)=shape(Vc)
    neq=(n1**2-n1)/2+1
    Lmat=zeros((neq,n1))*1.
    Rvec=zeros(neq)*1.
    cnt=0
    for i in range(n1):
        for j in range(i):
            Lmat[cnt,i]=1.
            Lmat[cnt,j]=-1.
            if(trialE[i] > trialE[j]):
                Rvec[cnt]=deltaE(i,j,Vc,trialE,kbT)
            else:
                Rvec[cnt]=deltaE(i,j,Vc,trialE,kbT)
            cnt=cnt+1
    Lmat[cnt,1]=1.
    Rvec[cnt]=trialE[1]*Hcm


    (deltaEmin,res,rank,s)=linalg.lstsq(Lmat,Rvec)
    return (deltaEmin,res,rank,s)

def solveEpairs(Vc,trialE,kbT,Hcm,pairlist):
    (n1,n2)=shape(Vc)
    neq=len(pairlist)+1
    Lmat=zeros((neq,n1))*1.
    Rvec=zeros(neq)*1.
    cnt=0
    for pair in pairlist:
        i=pair[0]#-1
        j=pair[1]#-1
        Lmat[cnt,i]=1.
        Lmat[cnt,j]=-1.
        #if(trialE[i] > trialE[j]):
        Rvec[cnt]=deltaE(i,j,Vc,trialE,kbT)-(trialE[i]-trialE[j])*Hcm
        print "delta E",deltaE(i,j,Vc,trialE,kbT),(trialE[i]-trialE[j])*Hcm
        #else:
        #    Rvec[cnt]=deltaE(i,j,Vc,trialE,kbT)-(trialE[i]-trialE[j])*Hcm
        cnt=cnt+1
    Lmat[cnt,1]=1.
    Rvec[cnt]=0.#trialE[1]*Hcm
    
    (deltaEmin,res,rank,s)=linalg.lstsq(Lmat,Rvec)
    print "deltaEmin",deltaEmin
    deltaEmin=trialE+deltaEmin/Hcm
    return (deltaEmin,res,rank,s)

def solveEpairs_Ebar(Vc,trialE,kbT,Hcm,pairlist):
    (n1,n2)=shape(Vc)
    ebarcol=n1
    newE=zeros(n1)
    neq=len(pairlist)+n1
    Lmat=zeros((neq,n1+1))*1.
    Rvec=zeros(neq)*1.
    cnt=0
    for pair in pairlist:
        i=pair[0]#-1
        j=pair[1]#-1
        Lmat[cnt,i]=1.
        Lmat[cnt,j]=-1.
        if(trialE[i] > trialE[j]):
            Rvec[cnt]=deltaE(i,j,Vc,trialE,kbT)/Hcm
        else:
            Rvec[cnt]=deltaE(i,j,Vc,trialE,kbT)/Hcm
        cnt=cnt+1
    for i in range(n1):
        Lmat[cnt,i]=1
        Lmat[cnt,ebarcol]=1
        Rvec[cnt]=trialE[i]
        cnt=cnt+1

    (soln,res,rank,s)=linalg.lstsq(Lmat,Rvec)
    for i in range(n1):
        newE[i]=soln[-1]+soln[i]
    return (newE,res,rank,s)

def permsolveE(Vc,trialE,perm,kbT,Hcm):
    (n1,n2)=shape(Vc)
    neq=(n1**2-n1)/2+1
    Lmat=zeros((neq,n1))
    Rvec=zeros(neq)
    cnt=0
    for i in range(n1):
        for j in range(i):
            Lmat[cnt,i]=1.
            Lmat[cnt,j]=-1.
            if(perm[i] > perm[j]):
                Rvec[cnt]=abs(deltaE(i,j,Vc,trialE,kbT))
            else:
                Rvec[cnt]=-abs(deltaE(i,j,Vc,trialE,kbT))
            cnt=cnt+1
    Lmat[cnt,0]=1.
    Rvec[cnt]=0.#trialE[0]*Hcm


    (deltaEmin,res,rank,s)=linalg.lstsq(Lmat,Rvec)
    return (deltaEmin,res,rank,s)

def leastResSolve(Vc,trialE,kbT,Hcm):
    nstates=len(trialE)
    order=range(nstates)
    perms=list(itertools.permutations(order))
    leastres=1.e10
    for perm in perms:
        (deltaEmin,res,rank,s)=permsolveE(Vc,trialE,perm,kbT,Hcm)
        if(res < leastres):
            lrdE=deltaEmin
            leastres=res
            lrrank=rank
            lrs=s
            lrperm=perm
    return (lrdE,leastres,lrrank,lrs,lrperm)

def iterateE(Vc,trialE,kbT,Hcm):
    imax=10
    frac=1./100.
    for i in range(imax):
        (dEnew,res,rank,s)=solveE(Vc,trialE,kbT,Hcm)
        print "res ",res
        #print "dEnew",dEnew
        dEnew=dEnew/Hcm
        #print "trialE",trialE
        trialE=(trialE+dEnew)/2.#trialE+frac*(dEnew-trialE)
    return dEnew

def Edist(E1,E2):
    n1=len(E1)
    dist=0.
    for i in range(n1):
        dist=dist+abs(E1[i]-E2[i])
    return dist/n1

def permEdist(E1,E2):
    n1=len(E1)
    perms=list(itertools.permutations(E1))
    #nperms=list(itertools.permutations(range(n1)))
    mindist=1e11
    for perm in perms:
        dist=Edist(perm,E2)
        if(dist < mindist):
            mindist = dist
            minperm=perm
    return (mindist,minperm)

def constpairlist(Vc,trialE,kbT,Hcm,rmin,rmax):
    pairlist=[]
    (n1,n2)=shape(Vc)
    for i in range(n1):
        for j in range(i):
            if(i !=j):
                #print "i,j",i,j,(abs(deltaE(i,j,Vc,trialE,kbT))/(abs(trialE[i]-trialE[j])*Hcm))
                ratio=(abs(deltaE(i,j,Vc,trialE,kbT))/(abs(trialE[i]-trialE[j])*Hcm))
                if((rmin<ratio) and (ratio< rmax)):
                    pairlist.append((i,j))
    return pairlist

def constpairlist_ediff(Vc,trialE,kbT,Hcm,ediff):
    pairlist=[]
    (n1,n2)=shape(Vc)
    for i in range(n1):
        for j in range(i):
            if(i !=j):
                #print "i,j",i,j,(abs(deltaE(i,j,Vc,trialE,kbT))/(abs(trialE[i]-trialE[j])*Hcm))
                dE=abs((abs(deltaE(i,j,Vc,trialE,kbT))-(abs(trialE[i]-trialE[j])*Hcm)))
                #print "dE",i,j,dE
                if(dE<ediff):
                    pairlist.append((i,j))
    return pairlist


def energyratios(Vc,trialE,kbT,Hcm):
    (n1,n2)=shape(Vc)
    ratarray=zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            ratarray[i][j]=(abs(deltaE(i,j,Vc,trialE,kbT))/(abs(trialE[i]-trialE[j])*Hcm))
            #ratarray[i][j]=abs((abs(deltaE(i,j,Vc,trialE,kbT))-(abs(trialE[i]-trialE[j])*Hcm)))
            #ratarray[i][j]=abs(Vc[i,j])*exp(-abs(trialE[i]-trialE[j])/(2.*kbT))/(4*kbT)
        ratarray[i][i]=0.
    return ratarray

def energydiffs(Vc,trialE,kbT,Hcm):
    (n1,n2)=shape(Vc)
    ratarray=zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            #ratarray[i][j]=abs((abs(deltaE(i,j,Vc,trialE,kbT))-(abs(trialE[i]-trialE[j])*Hcm)))
            ratarray[i][j]=(((deltaE(i,j,Vc,trialE,kbT))-((trialE[i]-trialE[j])*Hcm)))
            
        ratarray[i][i]=0.
    return ratarray


def Vcratios(Vc,trialE,kbT,Hcm):
    (n1,n2)=shape(Vc)
    ratarray=zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            #ratarray[i][j]=(abs(deltaE(i,j,Vc,trialE,kbT))/(abs(trialE[i]-trialE[j])*Hcm))
            
            #print i,j,log(4.*abs(Vc[i,j])/kbT),-abs((trialE[i]-trialE[j])/(2.*kbT))
            ratarray[i][j]=4.*abs(Vc[i,j])*exp(-abs((trialE[i]-trialE[j])/(2.*kbT)))
            #ratarray[i][j]=abs(Vc[i,j])*exp(-abs(deltaE(i,j,Vc,trialE,kbT)/Hcm)/(2.*kbT))
#deltaE(i,j,Vc,trialE,kbT)
            ratarray[i][j]=ratarray[i][j]/(kbT)
        ratarray[i][i]=0.
    return ratarray

temp=300.#300.#300.
#(newE,res,rank,s)=solveE(Vc,trialE,kboltz*temp,Hcm)
##print "newE",newE
#print "trialE",trialE*Hcm
##newE=iterateE(Vc,trialE,kboltz*temp,Hcm)
#newE=newE+average(trialE*Hcm-newE)
#print "res",res
#print "avg shift",average(abs(newE-trialE*Hcm))
#print "avg sq shift",sqrt(average(abs(newE-trialE*Hcm)**2.))
#print "newE",newE
#print "Edist",Edist(newE,trialE*Hcm)
#print "permEdist",permEdist(newE,trialE*Hcm)
#print ""

#pairlist=[(1,3),(1,4),(1,7),(1,8),(2,3),(2,4),(2,6),(2,7),(3,4),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7),(4,8),(5,6),(5,7),(6,7),(6,8)]
#pairlist=[(1,4),(1,5),(1,7),(1,8),(2,8),(3,7),(4,6),(4,8),(6,8)]
rat=1.2
#pairlist=constpairlist(Vc,trialE,kboltz*temp,Hcm,1/rat,rat)
pairlist=constpairlist_ediff(Vc,trialE,kboltz*temp,Hcm,kboltz*temp*Hcm/2.)
#pairlist=[(2,3)]#test
print "pairlist",len(pairlist),pairlist
ratarray=energydiffs(Vc,trialE,kboltz*temp,Hcm)
print "ratarray",ratarray
Vcarray=Vcratios(Vc,trialE,kboltz*temp,Hcm)
print "Vcarray",Vcarray[3]
#(newE,res,rank,s)=solveEpairs(Vc,trialE,kboltz*temp,Hcm,pairlist)
#(newE,res,rank,s)=solveEpairs_Ebar(Vc,trialE,kboltz*temp,Hcm,pairlist)
(newE,res,rank,s)=solveEpairs(Vc,trialE,kboltz*temp,Hcm,pairlist)
##print "corrections",newE
##newE=trialE+newE/Hcm
##print "newE",newE
#print "trialE",trialE*Hcm
#print "newE",newE*Hcm
##newE=iterateE(Vc,trialE,kboltz*temp,Hcm)
##newE=newE+average(trialE*Hcm-newE)
#print "res",res
#print "avg shift",average(abs(newE*Hcm-trialE*Hcm))
#print "avg sq shift",sqrt(average(abs(newE*Hcm-trialE*Hcm)**2.))
#print "newE",newE
#print "Edist",Edist(newE*Hcm,trialE*Hcm)
#print "permEdist",permEdist(newE*Hcm,trialE*Hcm)
#newpairlist=constpairlist(Vc,newE,kboltz*temp,Hcm,1/rat,rat)
#print "newpairlist",len(newpairlist),newpairlist
ratarray=energydiffs(Vc,newE,kboltz*temp,Hcm)#Vcratios(Vc,newE,kboltz*temp,Hcm)#energydiffs(Vc,newE,kboltz*temp,Hcm)
print "new ratarray",ratarray[3]
#
Vcarray=Vcratios(Vc,newE,kboltz*temp,Hcm)
print "Vcarray",Vcarray[3]

#trialEp=zeros(len(trialE))
#trialEp=trialE
#for i in range(100):
#    #pairlist=constpairlist(Vc,trialEp,kboltz*temp,Hcm,1/rat,rat)
#    pairlist=constpairlist_ediff(Vc,trialEp,kboltz*temp,Hcm,kboltz*temp*Hcm/2.)
#    (newE,res,rank,s)=solveEpairs(Vc,trialEp,kboltz*temp,Hcm,pairlist)
#    trialEp=newE
#    #print "pairs", len(pairlist)
#    #print "newE",i,newE*Hcm
#    #print "Edist",Edist(newE*Hcm,trialE*Hcm)
#    ratarray=energydiffs(Vc,newE,kboltz*temp,Hcm)
#print "pairlist",len(pairlist),pairlist
#print "ratarray",ratarray
#print "Edist",Edist(newE*Hcm,trialE*Hcm)

#print "newpairlist",constpairlist(Vc,trialE,kboltz*temp,Hcm,.5,2.)
#(newE,res,rank,s,perm)=leastResSolve(Vc,trialE,kboltz*temp,Hcm)
#print "lrperm",perm
#print "newE",newE
#print "res",res
#newE=newE+average(trialE*Hcm)-average(newE)
#print "res",res
#print "avg shift",average(abs(newE-trialE*Hcm))
#print "avg sq shift",sqrt(average(abs(newE-trialE*Hcm)**2.))
#print "newE",newE
#print "trialE",trialE*Hcm
#print "Edist",Edist(newE,trialE*Hcm)
#print "permEdist",permEdist(newE,trialE*Hcm)
