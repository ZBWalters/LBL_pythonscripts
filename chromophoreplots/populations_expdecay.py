from numpy import *
from scipy import *
import scipy.integrate
#from itertools import *
import itertools
from pylab import *

Hnm=45.565# nm = Hnm/Estate
Hcm=2.1947e5
kboltz=3.16697647924e-6
aut=2.418884326505e-17

######ps2 rc
#Vc=[[0,150,0,-42,-6,17],[150,0,-60,-36,21,-3],[-42,-60,0,7,47,-4],[-53,-36,7,0,-5,35],[-6,21,47,-5,0,3],[17,-3,-4,35,3,0]]
#trialE=[666,666,678,667,672,675]
#trialE=(45.565/array(trialE))*Hcm

####fmo
###mpi values

#old values
#Vc=[[0,94.8,5.5,-5.9,7.1,-15.1,-12.2,39.5],[94.8,0,29.8,7.6,1.6,13.1,5.7,7.9],[5.5,29.8,0,-58.9,-1.2,-9.3,3.4,1.4],[-5.9,7.6,-58.9,0,-64.1,-17.4,-62.3,-1.6],[7.1,1.6,-1.2,-64.1,0,89.5,-4.6,4.4],[-15.1,13.1,-9.3,-17.4,89.5,0,35.1,-9.1],[-12.2,5.7,3.4,-62.3,-4.6,35.1,0,-11.7],[39.5,7.9,1.4,-1.6,4.4,-9.1,-11.7,0]]
#trialE=[12505,12425,12195,12375,12600,12515,12465,12700]

#current values
#Vc=[[0,-87.7,5.5,-5.9,6.7,-13.7,-9.9,39.5],[-87.7,0,30.8,8.2,.7,11.8,4.3,2.4],[5.5,30.8,0,-53.5,-2.2,-9.6,6,2.4],[-5.9,8.2,-53.5,0,-70.7,-17,-63.3,2.4],[6.7,.7,-2.2,-70.7,0,81.1,-1.3,2.4],[-13.7,11.8,-9.6,-17,81.1,0,39.7,2.4],[-9.9,4.3,6,-63.3,-1.3,39.7,0,2.4],[39.5,2.4,2.4,2.4,2.4,2.4,2.4,0]]
#trialE=[12410,12530,12210,12320,12480,12630,12440,12700]


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


Etmp=[ 18766.9354248,   18525.81848145,  18485.63232422,  18452.14385986,
  18900.88928223,  19329.54162598,  19048.23852539,  19048.23852539]
trialE=E#E42p#47#E42#E47


##setting up parameters for main program

Vc=array(Vc)/Hcm
trialE=array(trialE)/Hcm

temp=77.#300.#300.
kbT=temp*kboltz
Gamma=kbT
lightspeed=29979245800#cm/sec

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

def defineddecaytimes_pairs(Vc,trialE,pairtemplist):
    (n1,n2)=shape(Vc)
    npair=len(pairtemplist)
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
    print "res",res
    newE=soln
    return newE

def leastres_timetemppairs(Vc,trialE,pairdecays):
    Eperms=list(itertools.permutations(trialE))
    minres=1e100
    for Eperm in Eperms:
        (newE,res)=defineddecaytimesandtemps(Vc,trialE,pairdecays)
        if(res < minres):
            minres=res
            minresnewE=newE
    return newE

def defineddecaytimesandtemps(Vc,trialE,pairdecays):
    (n1,n2)=shape(Vc)
    npair=len(pairdecays)
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
        lmbda=1./(lmbda*1.e-12/aut)
        kbT=pairdecays[i][2]*kboltz
        #print "kbT",i,kbT
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
    #print "rank",rank
    #print "s",s
    #print "res",res*Hcm
    #print "soln",soln
    newE=soln
    return (newE,res)


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

def dchrompopsdt(y,t0,lmat):
    nstates=len(y)
    dydt=y*0.
    for i in range(nstates):
        for j in range(nstates):
            diff=(y[i]-y[j])/2.
            dydt[i]=dydt[i]+lmat[i,j]*diff
            dydt[j]=dydt[j]-lmat[i,j]*diff
    return dydt

def dchrompopsdt2(t0,y,lmat):
    nstates=len(y)
    dydt=y*0j
    #print "lmat",lmat
    #if ybar=(y1+y2)/2, deltay=(y1-y2)/2
    #and d(deltay)/dt=-Gamma*deltay
    #then d(y1)/dt=-d(y2)/dt=-Gamma/2*(y1-y2)
    for i in range(nstates):
        for j in range(i):
            diff=(y[i]-y[j])/2.
            dydt[i]=dydt[i]+lmat[i,j]*diff
            dydt[j]=dydt[j]-lmat[i,j]*diff
    return dydt

def zodeint(tarray,y0,func,argtuple):
    #acts as a wrapper to scipy.integrate.ode
    tyarray=[y0]
    r=scipy.integrate.ode(func).set_integrator('zvode',method='bdf',with_jacobian=False)
    r.set_f_params(argtuple)
    r.set_initial_value(y0,tarray[0])
    for i in range(1,len(tarray)):
        #print "tarray[i]",i,tarray[i]
        r.integrate(tarray[i])
        tyarray.append(r.y)
    return array(tyarray)

def oneDarray2string(array):
    string=""
    for elt in array:
        string=string+str(elt)+" "
    return string

def plotpops(tarray,pvst):
    (nt,nstates)=shape(pvst)
    figure()
    for i in range(nstates):
        plot(tarray*aut,pvst[:,i],label=str(i))
    show()

def reweight_pops(kbT,trialE,pops):
    totpop=0.
    minE=min(trialE)
    Etmp=trialE-minE
    retpops=pops*0.
    for i in range(len(trialE)):
        retpops[i]=pops[i]*exp(-Etmp[i]/(kbT))
    totpop=sum(retpops)
    #print "totpop",totpop
    retpops=retpops/totpop
    return retpops

def reweight_amps(kbT,trialE,amps):
    totpop=0.
    minE=min(trialE)
    Etmp=trialE-minE
    retamps=amps*0.
    totpop=0.
    for i in range(len(trialE)):
        retamps[i]=amps[i]*exp(-Etmp[i]/(2*kbT))
        totpop=totpop+retamps[i]**2.
    #print "totpop",totpop
    retamps=retamps/sqrt(totpop)
    return retamps


def eigensystem(Veff,trialE):
    (n1,n2)=shape(Veff)
    Heff=zeros((n1,n2))
    Heff[:,:]=Veff[:,:]
    for i in range(n1):
        Heff[i,i]=trialE[i]
    w,vr=linalg.eig(Heff)
    return w,vr

def lambdamat_changebasis(lambdamat,M):
    (n1,n2)=shape(lambdamat)
    l2=zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            l2[i,j]=-lambdamat[i,j]/2
        l2[i,i]=sum(lambdamat[i,:])/2
    l2p=dot(M,dot(l2,transpose(M)))
    print "l2p",l2p
    lret=zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            lret[i,j]=l2p[i,j]*2
    return lret

def ratematch(array,rate,factor):
    pairlist=[]
    ratelist=[]
    (n1,n2)=shape(array)
    for i in range(n1):
        for j in range(i,n2):
            if((abs(array[i,j])/factor < rate) and (abs(array[i,j])*factor > rate)):
                pairlist.append((i,j))
                ratelist.append(array[i,j])
    return pairlist,ratelist

def bestfitrate(array,rate):
    (n1,n2)=shape(array)
    logbestfit=Inf
    bestfitpair=[]
    for i in range(n1):
        for j in range(i,n2):
            if(abs(array[i,j])==0):
                logratio=Inf
            else:
                logratio=abs(log(rate)-log(abs(array[i,j])))
            if(logratio<logbestfit):
                logbestfit=logratio
                bestfitpair=(i,j)
    return bestfitpair,exp(logbestfit)

def twoDarray2string(array):
    (n1,n2)=shape(array)
    #print "n1,n2",(n1,n2)
    string=""
    for i in range(n1):
        for j in range(n2):
            string=string+str(array[i,j])+"\t"
    string=string+"\n"
    return string

def term2latex(term):
    retstr='%.3g' % abs(real(term))
    if abs(imag(term)) > 0:
        retstr=retstr+' + '+ '%.3g' % abs(imag(term))+'i'
    if(abs(term) == 0):
        retstr=" "
    return retstr

def row2latex(row):
    n=len(row)
    retstr=term2latex(row[0])
    for i in range(1,n):
        retstr=retstr+' & '+term2latex(row[i])
    return retstr

def array2latex(array):
    (n1,n2)=shape(array)
    lbls=[r'$\beta_{50C}$', r'$\alpha_{19A}$' , r'$\alpha_{19B}$' , r'$\beta_{82C}$' , r'$\beta_{158C}$' , r'$\beta_{50D}$' , r'$\beta_{82D}$' , r'$\beta_{158D}$']
    retstr=""
    for lbl in lbls:
        retstr=retstr+" & "+lbl
    retstr=retstr+r" \\ "
    for i in range(n1):
        retstr=retstr+lbls[i]+" & "+row2latex(array[i,:])+r' \\ '
    return retstr

def popsvst2file(nchrom,nsteps,deltat,kbT,trialE,lmat):
    nstates=len(trialE)
    chrompops=zeros(nstates)*1j

    chrompops[nchrom-1]=1.
    chrompops=array(chrompops)
    chrompops=reweight_pops(-kbT,trialE,chrompops)
    tarray=array(range(2001))/2001.*deltat

    #print "tarray",tarray
    #print "aut",aut
    #print "lmat",lmat
    chrompops_vs_t=zodeint(tarray,chrompops,dchrompopsdt2,lmat)
    #print "chrompops_vs_t",chrompops_vs_t
    
    #To plot the file
    plotpops(tarray,real(chrompops_vs_t))

    filename="expdecay.p"+str(nchrom)+str(nchrom)+".dat"
    print "filename",filename
    outfile=open(filename,'w')
    for i in range(len(tarray)):
        outfile.write(str(tarray[i]*aut)+"\t"+ oneDarray2string(real(chrompops_vs_t[i]))+"\n")  
    outfile.close()
    return

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

#calculate decay times in picoseconds
psmat=1./lslow*aut/1e-12
#psmat=matcutoff(psmat,200.)
print "psmat",abs(psmat)
#calculate rate of pop transfer between eigenstates of Veffmat
#evals,evecs=eigensystem(Veffmat,trialE)
#lnew=lambdamat_changebasis(lslow,evecs)
#print "lnew",lnew
#psmatp=1./lnew*aut/1e-12
#psmatp=matcutoff(psmatp,100.)
#for i in range(len(evals)):
#    evecs[i,:]=reweight_amps(kbT,trialE,evecs[i,:])

nstates=len(trialE)
chrompops=zeros(nstates)*1j

#popsvst2file(1,2001,1e-15/aut,kbT,trialE,lslow)
for i in range(1,9):
    popsvst2file(i,2001,10.e-12/aut,kbT,trialE,lslow)

##print lslow to a file
outfile=open('transferrates.dat','w')
outfile.write(twoDarray2string(real((psmat))))
outfile.close()

