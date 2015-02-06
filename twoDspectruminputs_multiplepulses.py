import sys
import glob
import os
import stat
import subprocess
import shutil
from numpy import pi
from numpy import *
from multiprocessing import *
from multiprocessing import Pool

#nprocs=int(sys.argv[-1])
#print("nprocs\t"+str(nprocs))

aut=24.2#attoseconds per atomic unit of time
Hrt=27.21

#this script constructs a master bash script and all input files for
#one calculation run as used in a multidimensional spectroscopy
#calculation.  Component calculations will be set up by replacing the
#appropriate variables in template files located in templates/  

#The full calculation will be run in its own directory, so that the
#mctdhf code can be run in parallel without trying to read/write the
#same files

#The version of this script used on NERSC computers does not
#execute the calculations, but rather prepares a jobfile which
#can be read by taskfarmer.

def dictionaryreplace(text,dictionary):
    newtext=text
    for i,j in dictionary.items():#.iteritems():
        newtext=newtext.replace(i,j)
    return newtext


#def constructdictionary(n):
#    w=(2*pi)/(400/aut)
#    omegaval=str(w)#2.*pi/8. #frequency of sin^2 envelope
#    omega2val="1.3d0" #frequency of driving laser
#    ionpulseintensityval="1.d-1"
#    measurementpulseintensityval="1.d-3"
#
#    totaldelay=5.1
#    delaytimestep=.1
#    propagationtimestep=str(0.05)
#    t1val=str(2*n*delaytimestep)
#    t2val=str(totaldelay-n*delaytimestep)
#    suffix='.'+str(n)
#
#    dict={'#omega#':omegaval, '#omega2#':omega2val, '#ionpulseintensity#':ionpulseintensityval, '#measurementpulseintensity#':measurementpulseintensityval, '#t1#':t1val, '#t2#':t2val, '#suffix#':suffix, '#timestep#':propagationtimestep}
#    return dict

def pulsedictionary(w,w2,intensity,phaseshift,timestep):
    dict={'#omega#':str(w), '#omega2#':str(w2), '#intensity#':str(intensity),'#phaseshift#':str(phaseshift), '#timestep#':str(timestep)}
    return dict

def delaydictionary(tdelay,timestep):
    dict={'#tdelay#':str(tdelay), '#timestep#':str(timestep)}
    return dict

def inpoutdictionary(inpstr,outstr):
    dict={'#inpstr#':inpstr,'#outstr#':outstr}
    return dict

def filedictionary(inpfilename,outfilename):
    dict={'#inpfilename#':inpfilename,'#outfilename#':outfilename}
    return dict

def measurementdictionary(measurementtimestep):
    dict={'#measurementtimestep#':str(measurementtimestep)}
    return dict

###################
def templatereplace(oldfilename,newfilename,dictlist):
    oldfile=open(oldfilename)
    newfilestring=oldfile.read()
    
    for dict in dictlist:
        newfilestring=dictionaryreplace(newfilestring,dict)
        
    
    newfile=open(newfilename,'w+')
    newfile.write(newfilestring)
    #os.chmod(newfilename,stat. S_IXUSR)
    oldfile.close()
    newfile.close()
    
def stepstr(n):
    return "step_"+str(n)

def maketstartlist(deltatlist,wenvlist):
    tstartlist=[0.]#first pulse starts at time 0
    for i in range(len(deltatlist)):
        oldtmid=tstartlist[-1]+(pi/2)/wenvlist[i]
        newtmid=oldtmid+deltatlist[i]
        newtstart=newtmid-(pi/2)/wenvlist[i+1]#note that wenvlist is 1
                                              #longer than deltatlist
        tstartlist.append(newtstart)
    return tstartlist


def initialrelaxation(masterscriptfile,dirstr):
    bashtemplate="templates/Relax.Bat.template"
    inputtemplate="templates/Input.Inp.Relax.template"

    steproot=stepstr(0)
    
    bashfilename=steproot+".Bat"
    inpfilename="Input.Inp."+steproot
    outfilename="Out.States."+steproot

    #list of replacement rules
    dictlist=[inpoutdictionary(steproot,steproot),filedictionary(inpfilename,outfilename)]

    #use dictlist to convert template files to usable input & script files
    templatereplace(bashtemplate,dirstr+bashfilename,dictlist)
    templatereplace(inputtemplate,dirstr+inpfilename,dictlist)

    masterscript=open(dirstr+masterscriptfile,'w')
    masterscript.write("bash "+bashfilename+"\n")
    masterscript.close()

def excitationpulse(masterscriptfile,dirstr,stepindx,w,w2,intensity,phaseshift,timestep):
    inpstr=stepstr(stepindx-1)
    outstr=stepstr(stepindx)
    steproot=stepstr(stepindx)

    bashtemplate="templates/Excitationpulse.Bat.template"
    inputtemplate="templates/Input.Inp.Excitationpulse.template"

    bashfilename=steproot+".Bat"
    inpfilename="Input.Inp."+steproot
    outfilename="Out.States."+steproot

    #list of replacement rules
    dictlist=[inpoutdictionary(inpstr,outstr),filedictionary(inpfilename,outfilename),pulsedictionary(w,w2,intensity,phaseshift,timestep)]

    #use dictlist to convert template files to usable input & script files
    templatereplace(bashtemplate,dirstr+bashfilename,dictlist)
    templatereplace(inputtemplate,dirstr+inpfilename,dictlist)

    masterscript=open(dirstr+masterscriptfile,'a')
    masterscript.write("bash "+bashfilename+"\n")
    masterscript.close()

def multiplepulses(masterscriptfile,dirstr,stepindx,wlaslist,wenvlist,Ilist,CEPlist,tstartlist,timestep):
    
    inpstr=stepstr(stepindx-1)
    outstr=stepstr(stepindx)
    steproot=stepstr(stepindx)

    bashtemplate="templates/Multiplepulses.Bat.template"
    inputtemplate="templates/Input.Inp.Multiplepulses.template"

    bashfilename=steproot+".Bat"
    inpfilename="Input.Inp."+steproot
    outfilename="Out.States."+steproot


    #make strings from the various input lists
    wlasstr=",".join(map(str,wlaslist))
    wenvstr=",".join(map(str,wenvlist))
    Istr=",".join(map(str,Ilist))
    CEPstr=",".join(map(str,CEPlist))
    tstartstr=",".join(map(str,tstartlist))
    numpulses=len(wlaslist)
    pulsetypestr="2"
    for i in range(numpulses-1):
        pulsetypestr+=", 2"
        
    multiplepulsedict={"#numpulses#":str(numpulses), "#pulsetypelist#":pulsetypestr, "#omega2list#":wlasstr, "#omegalist#":wenvstr, "#intensitylist#":Istr, "#CEPlist#":CEPstr, "#pulsestartlist#":tstartstr, '#timestep#':str(timestep)}
    dictlist=[inpoutdictionary(inpstr,outstr),filedictionary(inpfilename,outfilename),multiplepulsedict]
    
    #use dictlist to convert template files to usable input & script files
    templatereplace(bashtemplate,dirstr+bashfilename,dictlist)
    templatereplace(inputtemplate,dirstr+inpfilename,dictlist)

    masterscript=open(dirstr+masterscriptfile,'a')
    masterscript.write("bash "+bashfilename+"\n")
    masterscript.close()

def measurementpulse(masterscriptfile,dirstr,stepindx,w,w2,intensity,phaseshift,timestep,measurementtimestep):
    inpstr=stepstr(stepindx-1)
    outstr=stepstr(stepindx)
    steproot=stepstr(stepindx)

    bashtemplate="templates/Measurementpulse.Bat.template"
    inputtemplate="templates/Input.Inp.Measurementpulse.template"

    bashfilename=steproot+".Bat"
    inpfilename="Input.Inp."+steproot
    outfilename="Out.States."+steproot

    #list of replacement rules
    dictlist=[inpoutdictionary(inpstr,outstr),filedictionary(inpfilename,outfilename),pulsedictionary(w,w2,intensity,phaseshift,timestep),measurementdictionary(measurementtimestep)]

    #use dictlist to convert template files to usable input & script files
    templatereplace(bashtemplate,dirstr+bashfilename,dictlist)
    templatereplace(inputtemplate,dirstr+inpfilename,dictlist)

    masterscript=open(dirstr+masterscriptfile,'a')
    masterscript.write("bash "+bashfilename+"\n")
    masterscript.close()

 
def interpulseinterval(masterscriptfile,dirstr,stepindx,tdelay,timestep):
    inpstr=stepstr(stepindx-1)
    outstr=stepstr(stepindx)
    steproot=stepstr(stepindx)

    bashtemplate="templates/Interpulse_Interval.Bat.template"
    inputtemplate="templates/Input.Inp.Interpulse_Interval.template"

    bashfilename=steproot+".Bat"
    inpfilename="Input.Inp."+steproot
    outfilename="Out.States."+steproot

    #list of replacement rules
    dictlist=[inpoutdictionary(inpstr,outstr),filedictionary(inpfilename,outfilename),delaydictionary(tdelay,timestep)]

    #use dictlist to convert template files to usable input & script files
    templatereplace(bashtemplate,dirstr+bashfilename,dictlist)
    templatereplace(inputtemplate,dirstr+inpfilename,dictlist)

    masterscript=open(dirstr+masterscriptfile,'a')
    masterscript.write("bash "+bashfilename+"\n")
    masterscript.close()

def measurmentinterval(masterscriptfile,dirstr,stepindx,tdelay,timestep,measurementtimestep):
    inpstr=stepstr(stepindx-1)
    outstr=stepstr(stepindx)
    steproot=stepstr(stepindx)

    bashtemplate="templates/Measurementinterval.Bat.template"
    inputtemplate="templates/Input.Inp.Measurementinterval.template"

    bashfilename=steproot+".Bat"
    inpfilename="Input.Inp."+steproot
    outfilename="Out.States."+steproot

    #list of replacement rules
    dictlist=[inpoutdictionary(inpstr,outstr),filedictionary(inpfilename,outfilename),delaydictionary(tdelay,timestep),measurementdictionary(measurementtimestep)]

    #use dictlist to convert template files to usable input & script files
    templatereplace(bashtemplate,dirstr+bashfilename,dictlist)
    templatereplace(inputtemplate,dirstr+inpfilename,dictlist)

    masterscript=open(dirstr+masterscriptfile,'a')
    masterscript.write("bash "+bashfilename+"\n")
    masterscript.close()

        

#def fullcalculation(resultdir,calcindx,deltat1,deltat2,deltat3,CEP1,CEP2,CEP3,w,w2,I1,I2,I3,Imeasure,timestep,tmeasure,measurementtimestep):
def fullcalculation(resultdir,calcindx,deltatlist,CEPlist,wenvlist,wlaslist,\
                    Ilist,timestep,tmeasure,measurementtimestep):
    dirstr=resultdir+str(calcindx)+"/"
    subprocess.call(["mkdir",dirstr])
    masterscriptfile="fullcalculation.sh"

    [I1,I2,I3]=Ilist
    [CEP1,CEP2,CEP3]=CEPlist
    [wenv1,wenv2,wenv3]=wenvlist
    [wlas1,wlas2,wlas3]=wlaslist
    tstartlist=maketstartlist(deltatlist,wenvlist)

    #call subroutines to setup each step of the calculation
    initialrelaxation(masterscriptfile,dirstr)
    multiplepulses(masterscriptfile,dirstr,1,wlaslist,wenvlist,Ilist,CEPlist,tstartlist,timestep)
#    excitationpulse(masterscriptfile,dirstr,1,wenv1,wlas1,I1,CEP1,timestep)
#    interpulseinterval(masterscriptfile,dirstr,2,deltat1,timestep)
#    excitationpulse(masterscriptfile,dirstr,3,wenv2,wlas2,I2,CEP2,timestep)
#    interpulseinterval(masterscriptfile,dirstr,4,deltat2,timestep)
#    excitationpulse(masterscriptfile,dirstr,5,wenv3,wlas3,I3,CEP3,timestep)
#    interpulseinterval(masterscriptfile,dirstr,6,deltat3,timestep)
#    #measurementpulse(masterscriptfile,dirstr,7,w,w2,Imeasure,0.,timestep,measurementtimestep)
    measurmentinterval(masterscriptfile,dirstr,2,tmeasure,timestep,measurementtimestep)

    #execute masterscriptfile to actually perform the calculation
    currentdir=os.getcwd()
    #subprocess.call(["cd",dirstr])
    os.chdir(dirstr)
#    nodefile=os.environ['PBS_NODEFILE']
#    os.system("cat "+nodefile)
#    newnodefile="nodelist.txt"
#    makehostfile(newnodefile)
#    print("newnodefile\t"+newnodefile)
#    subprocess.call(['mpirun','-np','1','-hostfilefile',newnodefile,'-loadbalance','bash',masterscriptfile])#this is the problematic line
    #subprocess.call(['mpirun','-np','1','bash',masterscriptfile])
    #os.system("mpirun -np 1 -machinefile $PBS_NODEFILE --nooversubscribe -w bash "+masterscriptfile)#possible replacement
    os.chdir(currentdir)
    #subprocess.call(["cd",currentdir])
    
#############################
def printparameterlist(parameterlist,filename):
    f=open(filename,'w')
    f.write("#indx\tdeltat1\tdeltat2\tCEP1\tCEP2\tCEP3\n")
    for i in range(len(parameterlist)):
        [deltatlist,CEPlist]=parameterlist[i]
        printstr=str(i)+"\t"
        for a in range(len(deltatlist)):
            printstr+=str(deltatlist[a])+"\t"
        for a in range(len(CEPlist)):
            printstr+=str(CEPlist[a])+"\t"
        printstr+="\n"
        f.write(printstr)
    f.close()

def printparameterarray(paramarray,filename):
    f=open(filename,'w')
    #parameterarray[indx,0]=indx
    #parameterarray[indx,1]=deltat1array[i]
    #parameterarray[indx,2]=deltat2array[j]
    #parameterarray[indx,3]=CEParray[k][0]
    #parameterarray[indx,4]=CEParray[k][1]
    #parameterarray[indx,5]=CEParray[k][2]
    f.write("#indx\tdeltat1\tdeltat2\tCEP1\tCEP2\tCEP3\n")
    (n,m)=shape(paramarray)
    for i in range(n):
        f.write(str(i)+"\t"+str(paramarray[i,1])+"\t"+str(paramarray[i,2])+"\t"+str(paramarray[i,3])+"\t"+str(paramarray[i,4])+"\t"+str(paramarray[i,5])+"\n")
    f.close()

def processingrun(i):
    #deltat1=parameterarray[i][1]
    #deltat2=parameterarray[i][2]
    #deltat3=dt2#does not vary for 3 pulse setup
    #deltatlist=[deltat1,deltat2,deltat3]
    #CEP1=parameterarray[i][3]
    #CEP2=parameterarray[i][4]
    #CEP3=parameterarray[i][5]
    #CEPlist=[CEP1,CEP2,CEP3]
    [deltatlist,CEPlist]=parameterlist[i]

    fullcalculation(resultdir,i,deltatlist,CEPlist,wenvlist,wlaslist,\
                    Ilist,timestep,tmeasure,measurementtimestep)
    #fullcalculation(resultdir,i,deltat1,deltat2,deltat3,CEP1,CEP2,CEP3,w,w2,I1,I2,I3,Imeasure,timestep,tmeasure,measurementtimestep)

#    #write execute command in jobfile
#    jobfile=open("jobfile.txt","a")
#    jobfile.write("(cd results/"+str(i)+ " && exec bash fullcalcultion.sh)\n")

#############################
def makehostfile(hostfilename):
    outfile=open(hostfilename,'w')
    nodefile=os.environ['PBS_NODEFILE']
    nf=open(nodefile,'r')
    nodelist=nf.readlines()
    uniquenodes=list(set(nodelist))
    for node in uniquenodes:
        nnode=nodelist.count(node)
        outfile.write(node.strip()+" slots="+str(nnode)+"\n")
    outfile.close()
    nf.close()
###################################

#functions to choose dt,npts for desired energy resolution
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



####################################################################

#A multimensional calculation will involve a large number of parameter
#choices.  First, set up an array containing the full set of parameter
#choices, then iterate through this array for individual calculation
#runs.

#first, make a directory to hold all of the calculation results
resultdir="results_multiplepulse/"
subprocess.call(["mkdir",resultdir])

shutil.copy("./twoDspectruminputs_multiplepulses.py",resultdir)
#shutil.copy("./multiprocessingscript.py",resultdir)
#shutil.copy("./makejobfile.py",resultdir)
#shutil.copy("./taskfarmer.pbs",resultdir)
#shutil.copy("./taskfarmer.test.pbs",resultdir)


w2=7.884/Hrt#1s2s^2- 1s 2s^2 2p
w=w2/8.#((2*pi)/(1000/aut))#2./Hrt#((2*pi)/(1000/aut))#frequency of sin^2 envelope
exciteintensity=1.e-4
I1=exciteintensity
I2=exciteintensity
I3=exciteintensity
wlaslist=[w2,w2,w2]
wenvlist=[w,w,w]
Ilist=[I1,I2,I3]
timestep=.05#0.05

######################
#Setting up information for multiple calculations

dE=2.#.5#desired energy resolution in eV
maxE=16.#20.#desired maximum energy range in eV
dt1,T1max,ndt1=tparams(dE,maxE)
dt1=ceil(dt1/timestep)*timestep#make interpulse delays an integral
                               #number of timesteps
T1max=ndt1*dt1
deltat1array=dt1*array(range(1,ndt1+1))
#deltat1array=array([.1])


#dt2=.1
#ndt2=1
#deltat2array=dt2*array(range(1,ndt2+1))


#dt2,T2max,ndt2=tparams(dE,maxE)
#dt2=ceil(dt2/timestep)*timestep
#T2max=ndt2*dt2
#deltat2array=dt2*array(range(1,ndt2+1))
deltat2array=array([.1])#short delay between first two pulses and third pulse

dE2=2.
maxE2=16.
dt3,T3max,ndt3=tparams(dE2,maxE2)
dt3=ceil(dt3/timestep)*timestep#make interpulse delays an integral
                               #number of timesteps
T3max=ndt3*dt3

tmeasure=T3max#256#total interval to propagate when calculating dipole
measurementtimestep=dt3#4.#interval between calculating dipole



#CEParray=[[0,0,0]]
CEParray=[[0,0,0],[0,0,.5],[.5,0,1],[.5,0,.5],[1,0,0],[1,0,.5],[1.5,0,1.5],[0,0,1.5],[0,0,1],[.5,0,1.5],[1.5,0,1],[1.5,0,.5]]
#these choices of CEPs are taken from 
#Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)

ncalcs=len(deltat1array)*len(deltat2array)*len(CEParray)
parameterlist=[]
for i in range(len(deltat2array)):
    for j in range(len(deltat1array)):
        for k in range(len(CEParray)):
            deltatlist=[deltat1array[j],deltat2array[i]]
            CEPlist=CEParray[k]
            parameterlist.append([deltatlist,CEPlist])
printparameterlist(parameterlist,resultdir+"parameterkey.txt")


#parameterarray=zeros((ncalcs,6))
#indx=0
#for i in range(len(deltat2array)):
#    for j in range(len(deltat1array)):
#        for k in range(len(CEParray)):
#            parameterarray[indx,0]=indx
#            parameterarray[indx,1]=deltat1array[j]
#            parameterarray[indx,2]=deltat2array[i]
#            parameterarray[indx,3]=CEParray[k][0]
#            parameterarray[indx,4]=CEParray[k][1]
#            parameterarray[indx,5]=CEParray[k][2]
#            indx+=1
#
#printparameterarray(parameterarray,resultdir+"parameterkey.txt")
##savetxt(resultdir+"parameterkey.txt",parameterarray,delimiter='\t')


#os.system("rm jobfile.txt")
for i in range(16):
    processingrun(i)

###parallel execution of loop
#p=Pool(maxtasksperchild=1)
#p.map(processingrun,range(ncalcs),chunksize=1)#range(ncalcs),chunksize=1)#range(ncalcs))
#p.close()
#p.join

#for i in range(16):
#    processingrun(i)


