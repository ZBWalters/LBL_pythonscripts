from numpy import *
from scipy import *

Hnm=45.565# nm = Hnm/Estate
Hcm=2.1947e5
kboltz=3.16697647924e-6
aut=2.418884326505e-17
Hbyev=27.21
lightspeed=29979245800#cm/sec

temp=300.#300.#300.
kbT=temp*kboltz
Gamma=kbT



#######vibrational information

lambda0m=[40,70]
omegaj=[207,244,312,372,438,514,718,813,938,1111,1450,1520,1790,2090]
sj=[.0013,.0072,.0450,.0578,.0450,.0924,.0761,.0578,.0313,.0578,.1013,.0265,.0072,.0013]

lambda0m=array(lambda0m)
omegaj=array(omegaj)
sj=array(sj)

#PE545
E=[18532,18008,17973,18040,18711,19574,19050,18960]

Vc=[[0,1,-37,37,23,92,-16,12],[1,0,4,-11,33,-39,-46,3],[-37,4,0,45,3,2,-11,34],[37,-11,45,0,-7,-17,-3,6],[23,33,3,-7,0,18,7,6],[92,-39,2,-17,18,0,40,26],[-16,-46,-11,-3,7,40,0,7],[12,3,34,6,6,26,7,0]]

trialE=E

Vc=array(Vc)/Hcm
trialE=array(trialE)/Hcm

#####helper functions
def Veff(Vc,siteE,kbT):
    (n1,n2)=shape(Vc)
    Veff=zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            Veff[i,j]=Vc[i,j]*exp(-abs(siteE[i]-siteE[j])/(2.*kbT))
    for i in range(n1):
        Veff[i,i]=Veff[i,i]+siteE[i]
    return Veff


##main program
Veffmat=Veff(Vc,trialE,kbT)

(evals,evecs)= linalg.eig(Veffmat)
print "evals",evals*Hcm

print "evecs",evecs
(n1,n2)=shape(evecs)

lambdasum=sum(lambda0m[:])+sum(omegaj*sj)

minindex=list(evals).index(min(evals))
deltaE=zeros(n1)
#deltaE=deltaE+(evecs[:,minindex]**4)*lambdasum
reorgevals=evals
for i in range(n1):
    deltaE=sum(evecs[:,i]**4.)*2*lambdasum
    reorgevals[i]=evals[i]*Hcm-deltaE

hdiag=zeros((n1,n2))
for i in range(n1):
    hdiag[i,i]=reorgevals[i]
hnew=dot(transpose(evecs),dot(hdiag,evecs))
for i in range(n1):
    reorgevals[i]=hnew[i,i]

print "reorg evals",reorgevals
#for i in range(1):
#    print "cn^4",i,sum(evecs[:,i]**4)*lambdasum
#    deltaE[i]=sum(evecs[:,i]**4)*lambdasum

#print "E adjusted",trialE*Hcm-deltaE
#print "site repn", trialE*Hcm-lambdasum
