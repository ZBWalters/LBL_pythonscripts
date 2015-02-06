#This program calculates multidimensional spectra for three states in
#a lambda configuration as a way of investigating the lineshapes
from numpy import *
from numpy.linalg import solve
from scipy.integrate import ode

#constants
aut=24.2
Hrt=27.21



###########################################
###########################################
###########################################




def singlepulse(t,E0,wosc,wenv,t0,CEP):
    retval=0.
    #print("singlepulse\t"+str(t)+"\t"+str(t0)+"\t"+str(abs(t-t0)*wenv))
    if(abs(t-t0)*wenv<pi/2.):
        retval=E0*sin(wosc*(t-t0)+CEP)*pow(cos(wenv*(t-t0)),2)
    else:
        retval=0.
    return retval

def Elas(t,pulselist):
    Eret=0.
    for i in range(len(pulselist)):
        pulse=pulselist[i]
        #print("pulse in Elas\t"+str(i)+"\t"+str(pulse))
        Eret+=singlepulse(t,*pulse)
    return real(Eret)

def H(t,pulselist):
    retmat=H0+Elas(t,pulselist)*H1
#    retmat=H0*1.
#    #print("retmat H0\t"+str(retmat))
#    print("pulselist inside H(t)\t"+str(pulselist))
#    print("Elas\t"+str(Elas(t,pulselist)))
#    for i in range(len(pulselist)):
#        print("i\t"+str(i))
#        pulse=pulselist[i]
#        retmat+=singlepulse(t,*pulse)*H1
#        print("singlepulse\t"+str(singlepulse(t,*pulse)))
#        print("retmat\t"+str(retmat))
#    #print("shape H(t)\t"+str(shape(retmat)))
    return retmat

def dipexpect(psi):
    return dot(conjugate(transpose(psi)),dot(H1,psi))

def dpsidt(t,psi,pulselist):
    Ht=H(t,pulselist)
    #print("psi in dpsidt\t"+str(psi))
    #print("Ht in dpsidt\t"+str(Ht))
    retvec=1j*dot(H(t,pulselist),psi)
    #print("dpsidt\t"+str(retvec))
    #print("dpsidt shapes\t"+str(shape(Ht))+"\t"+str(shape(psi)))
    return retvec

def jacobian(t,psi,pulselist):
    return 1j*H(t,pulselist)

def makepulselist_dtcenter(dtlist,E0list,wosclist,wenvlist,CEPlist):
    #zeroth pulse begins at time t0
    pulselist=[]
    t0=pi/(2*wenvlist[0])
    pulselist.append([E0list[0],wosclist[0],wenvlist[0],t0,CEPlist[0]])
    for i in range(1,len(E0list)):
        oldt0=pulselist[i-1][3]
        newt0=oldt0+dtlist[i-1]
        pulselist.append([E0list[i],wosclist[i],wenvlist[i],newt0,CEPlist[i]])
    return pulselist

def makepulselist(dtlist,E0list,wosclis,wenvlist,CEPlist):
    #zeroth pulse begins at time t0
    pulselist=[]
    t0=pi/(2*wenvlist[0])
    pulselist.append([E0list[0],wosclist[0],wenvlist[0],t0,CEPlist[0]])
    for i in range(1,len(E0list)):
        oldt0=pulselist[i-1][3]
        oldwenv=wenvlist[i-1]
        wenv=wenvlist[i]
        newt0=oldt0+pi/(2*oldwenv)+(pi/2*wenv)+dtlist[i-1]
        pulselist.append([E0list[i],wosclist[i],wenvlist[i],newt0,CEPlist[i]])
    return pulselist

def dipevaltimes(lastt0,lastwenv,ndip,dtdip):
    return lastt0+pi/lastwenv+array(range(ndip))*dtdip

def controlledstep(r,dt,tf):
    tnext=r.t
    while(tnext<tf):
        tnext=min(r.t+dt,tf)
        r.integrate(tnext)

def dipolearray(psi0,pulselist,tarray):
    t0=0.
    r=ode(dpsidt,jacobian).set_integrator('zvode',method='bdf',
                                          with_jacobian=True)#,atol=1.e-10)
    r.set_initial_value(psi0,t0)
    r.set_f_params(pulselist).set_jac_params(pulselist)
    diparray=zeros(len(tarray))*0j
    dt=.1
    for i in range(len(tarray)):
        #r.integrate(tarray[i])
        controlledstep(r,dt,tarray[i])
        psi=r.y
        #print("t, psi\t"+str(tarray[i])+"\t"+str(psi))
        dipval=dipexpect(psi)
        #print("dipval\t"+str(dipval))
        diparray[i]=dipval

    return diparray

def finddipoles(psi0,pulselist,ndip,dtdip):
    lastt0=pulselist[-1][-2]
    lastwenv=pulselist[-1][-3]
    tarray=dipevaltimes(lastt0,lastwenv,ndip,dtdip)
    return dipolearray(psi0,pulselist,tarray)

def finddipoles_dtcenter(psi0,pulselist,dtdip,tmeasure):
    lastt0=pulselist[-1][-2]
    tstop=lastt0+tmeasure
    tarray=arange(0.,tstop,dtdip)
    return dipolearray(psi0,pulselist,tarray),tarray

def maxndip_dtcenter(wenv0,dtlist,dtdip,tmeasure):
    t0=(pi/2)/wenv0
    for i in range(len(dtlist)):
        t0+=dtlist[i]
    tstop=t0+tmeasure
    maxndip=int(ceil(tstop/dtdip))
    return maxndip

#def maxndip_dtcenter(pulselist,dtdip,tmeasure):
#    lastt0=pulselist[-1][-2]
#    tstop=lastt0+tmeasure
#    maxndip=floor(tstop/dtdip)

#################################
#phase matching routines
CEParray=[[0,0,0],[0,0,.5],[.5,0,1],[.5,0,.5],[1,0,0],[1,0,.5],[1.5,0,1.5],[0,0,1.5],[0,0,1],[.5,0,1.5],[1.5,0,1],[1.5,0,.5]]
#these choices of CEPs are taken from 
#Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)

Rvecarray=[[1,0,0],[0,1,0],[0,0,1],[1,-1,1],[1,1,-1],[-1,1,1],[2,-1,0],[2,0,-1],[-1,2,0],[0,2,-1],[-1,0,2],[0,-1,2]]
#Rvecarray taken from Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)
#Rvecarray[i].[k1,k2,k3]=k vector for phase matched component
        
def Linsys():
    Lmat=zeros((len(CEParray),len(Rvecarray)))*0j
    for i in range(len(CEParray)):
        for j in range(len(CEParray)):
            Lmat[i,j]=exp(1j*pi*dot(CEParray[i],Rvecarray[j]))
    return Lmat

def phasematchedcomponents(fulldiparray):
    (n1,n2,n3)=shape(fulldiparray)
    retdiparray=zeros((n1,n2,n3))*0j
    rhsarray=zeros((n3,n1*n2))*0j
    LHSmat=Linsys()
    for i in range(n1):
        for j in range(n2):
            indx=j+i*n2
            for k in range(n3):
                rhsarray[k,indx]=fulldiparray[i,j,k]
    solnarray=solve(LHSmat,rhsarray)
    for i in range(n1):
        for j in range(n2):
            indx=j+i*n2
            for k in range(n3):
                retdiparray[i,j,k]=solnarray[k,indx]
    return retdiparray

def arraytofile(xarray,yarray,zarray,filename):
    outfile=open(filename,'w+')
    (n,m)=shape(zarray)
    for i in range(n):
        for j in range(m):
            outfile.write(str(xarray[i])+"\t"+str(yarray[j])+"\t"+str(real(zarray[i,j]))+"\t"+str(imag(zarray[i,j]))+"\n")
    outfile.close()

#######################
###fourier transform routines
def FFTfreqarray(dt,npts):
    #nyquist critical frequency is 2 points per cycle
    #so that wcrit*(2*dt)=2*pi
    wcrit=2*pi/(dt)#2*dt
    dw=wcrit/npts
    return array(range(npts))*dw-wcrit/2

def twoDFFT(t1array,t2array,twoDdiparray):
    (np1,np2)=shape(twoDdiparray)
    print("twoDFFT shape twoDdiparray "+str(shape(twoDdiparray)))
    twodfreqarray=fft.fft2(twoDdiparray,(np1,np2),(0,1))
    twodfreqarray=fft.fftshift(twodfreqarray,axes=(0,1))
    print("twoDFFT shape twodfreqarray"+str(shape(twodfreqarray)))

    dt1=t1array[1]-t1array[0]
    dt2=t2array[1]-t2array[0]
    #fftfreq1=fft.fftfreq(np1,dt1)*2*pi
    #fftfreq2=fft.fftfreq(np2,dt2)*2*pi
    #fftfreq1=sort(fftfreq1)
    #fftfreq2=sort(fftfreq2)
    fftfreq1=FFTfreqarray(dt1,np1)
    fftfreq2=FFTfreqarray(dt2,np2)
    dw1=fftfreq1[1]-fftfreq1[0]
    dw2=fftfreq2[1]-fftfreq2[0]
    #print ("fftfreq1\t"+str(fftfreq1))
    #print ("fftfreq2\t"+str(fftfreq2))
    return [fftfreq1,fftfreq2,twodfreqarray]

def pulsefft(pulselist,dE,Emax):
    dt,Tmax,npts=tparams(dE,Emax)
    tarray=array(range(npts))*dt
    Elasarray=tarray*0.
    for i in range(len(tarray)):
        t=tarray[i]
        Elasarray[i]=Elas(t,pulselist)
        print ("t, Elas\t"+str(t)+"\t"+str(Elasarray[i]))
    fftfreq=FFTfreqarray(dt,npts)
    Efftarray=fft.fft(Elasarray)
    Efftarray=fft.fftshift(Efftarray)
    f=open("pulsefft.dat",'w')
    g=open("pulse.dat",'w')
    for i in range(len(tarray)):
        f.write(str(fftfreq[i]*Hrt)+"\t"+str(real(Efftarray[i]))+"\t"+str(imag(Efftarray[i]))+"\n")
        g.write(str(tarray[i])+"\t"+str(real(Elasarray[i]))+"\t"+str(imag(Elasarray[i]))+"\n")
        
    f.close()
############################################
#number of points, time spacing for a desired energy resolution
def tparams(dE, maxE):
    #dE is desired energy resolution, maxE is maximum energy we wish to resolve
    dEHrt=dE/Hrt
    maxEHrt=maxE/Hrt
    ErangeHrt=2*maxEHrt
    npts=int(floor(ErangeHrt/dEHrt))

    #critical sampling is 2 samples per period
    dt=(2*pi)/(ErangeHrt)
    Tmax=dt*npts
    return dt,Tmax,npts

def fftfreqcheck(dt,npts):
    return fftfreq(npts,dt)*2*pi*Hrt

def gibbsfilter(diparray,tarray,width):
    retdiparray=diparray*1.
    tlast=tarray[-1]
    (imax,jmax,kmax)=shape(diparray)
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                retdiparray[i,j,k]*=(1.-exp(-pow((tarray[j]-tlast)/width,2)))
    return retdiparray
########




###Main program
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################

E0=0
E1=5.3/Hrt#+.1/Hrt*1j
E2=10.884/Hrt#+.1/Hrt*1j
H0=diag([E0,E1,E2])

d1=1.
d2=1.
H1=array([[0,d1,d2],[d1,0,0],[d2,0,0]])


F0=0.01
wosc=(5.3+10.886)/(2*Hrt)#.55#.55
wenv=wosc/8.#(pi)/(1000/aut)#2./Hrt#(2*pi)/(1000/aut)#5.3/Hrt#.15205/2#(2*pi)/(1000/aut)

dEres=.5#energy resolution in eV
maxEres=15.#max energy resolution in eV
dt1,T1max,ndt1=tparams(dEres,maxEres)
dtdip,Tdipmax,ndip=tparams(dEres,maxEres)
dt1array=dt1*array(range(ndt1))
dtdiparray=dtdip*array(range(ndip))
dt2=.1#small value

##test:fourier transform pulse
##wenv=1/Hrt
#wenv=2./Hrt
#E0list=[0,F0,0]
#wosclist=[wosc,wosc,wosc]
#wenvlist=[wenv,wenv,wenv]
#CEPlist=[0,0,0]
#dtlist=[0.,0.]
#pulselist=makepulselist(dtlist,E0list,wosclist,wenvlist,CEPlist)
#pulsefft(pulselist,.001,30.)

#dtlist=[.1,.1]
E0list=[F0,F0,F0]
wosclist=[wosc,wosc,wosc]
wenvlist=[wenv,wenv,wenv]
#CEPlist=[0,0,0]

maxndip=maxndip_dtcenter(wenvlist[0],[max(dt1array),dt2],dtdip,Tdipmax)
fulldiparray=zeros((ndt1,maxndip,len(CEParray)))*0j
for i in range(len(dt1array)):
    for j in range(len(CEParray)):
        print("i,j\t"+str(i)+", "+str(j))
        dtlist=[dt1array[i],dt2]
        CEPlist=CEParray[j]
        pulselist=makepulselist_dtcenter(dtlist,E0list,wosclist,wenvlist,CEPlist)
        #print("pulselist\t"+str(pulselist))
        psi0=[1.,0j,0j]
        diparray,tarray=finddipoles_dtcenter(psi0,pulselist,dtdip,Tdipmax)
        ndip=len(diparray)
        fulldiparray[i,(maxndip-ndip):,j]=diparray[:]
        #for k in range(ndip):
        #    fulldiparray[i,(maxndip-ndip+k),j]=(diparray[k])*singlepulse(tarray[k],*pulselist[-1])
        #print("diparray\n"+str(diparray))

dtdiparray=array(range(maxndip))*dtdip
#fulldiparray=gibbsfilter(fulldiparray,dtdiparray,dtdip*50.)
phasematchedarray=phasematchedcomponents(fulldiparray)

#print phase matched dipoles to file
for phaseindx in range(len(CEParray)):
    filename="twoddip_"+str(phaseindx+1)+".dat"
    arraytofile(dt1array,dtdiparray,phasematchedarray[:,:,phaseindx],filename)


#two dimensional fourier transforms of phase matched dipoles
for phaseindx in range(len(CEParray)):
    filename="twodfreq_"+str(phaseindx+1)+".dat"
    [fftfreq1,fftfreq2,twodfreqarray]=twoDFFT(dt1array,dtdiparray,phasematchedarray[:,:,phaseindx])
    arraytofile(fftfreq1,fftfreq2,twodfreqarray,filename)
