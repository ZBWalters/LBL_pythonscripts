import sys
import glob
import os
import stat
import subprocess
import shutil
from numpy import pi
from numpy import *
from itertools import *
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

def nm_to_eV(nm):
    hc=1239.841#planck's constant in eV*nm
    eV=hc/nm
    return eV

def eV_to_nm(eV):
    hc=1239.841#planck's constant in eV*nm
    nm=hc/eV
    return nm

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
    dict={'#inpfilename#':inpfilename,'#outfilename#':outfilename,'#gridstr#':gridstring,'#atomstr#':atomstring}
    return dict

def measurementdictionary(measurementtimestep):
    dict={'#measurementtimestep#':str(measurementtimestep)}
    return dict

def CEPcenter(targetcenterphase,wlas,wenv):
    #MCTDHF code has pulse of the form A(t) sin(wenv t)^2
    #sin(wosc*t+startphase) however, it's often more convenient to
    #think about a pulse of the form 
    #A(t) cos(wenv(t-tmid))^2 cos(w(t-tmid)+centerphase).

    #This subroutine calculates startphase corresponding to a chosen
    #centerphase -- ie, so that 
    #sin(wosc*(t-tstart)+startphase)= cos(wosc*(t-tcenter)+centerphase),
    #where tcenter=tstart+(pi/2)/wenv
    
    tcenter=(pi/2)/wenv
    centerphase=tcenter*wlas
    startphase=mod(targetcenterphase-centerphase,2*pi)
    return startphase
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

def maketstartlist_streaking(deltatlist,wenvlist):
    tIR=0.# pulse starts at time 0
    tXUV=tIR+deltatlist[0]+(pi/2)/wenvlist[0]-(pi/2)/wenvlist[1]
    tstartlist=[tXUV,tIR]
    return tstartlist

def maketstartlist(deltatlist,wenvlist):
    tstartlist=[0.]#first pulse starts at time 0
    for i in range(len(deltatlist)):
        oldtmid=tstartlist[-1]+(pi/2)/wenvlist[i]
        #print("wenvlist[i]\t"+str(wenvlist[i]))
        #print("oldtmid/timestep\t"+str(oldtmid/timestep))
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

def multiplepulses(masterscriptfile, dirstr, stepindx, wlaslist, wenvlist,
                   Ilist, CEPlist, tstartlist, pulsethetalist, timestep,
                   measurementtimestep, measurementtime):
    
    inpstr=stepstr(stepindx-1)
    outstr=stepstr(stepindx)
    steproot=stepstr(stepindx)

    bashtemplate="templates/Multiplepulses.Bat.template"
    inputtemplate="templates/Input.Inp.Multiplepulses.template"

    bashfilename=steproot+".Bat"
    inpfilename="Input.Inp."+steproot
    outfilename="Out.States."+steproot

    #print("CEPlist\t"+str(CEPlist))
    startCEPlist=[]
    for i in range(len(CEPlist)):
        startCEP=CEPcenter(CEPlist[i]*pi,wlaslist[i],wenvlist[i])
        startCEPlist.append(startCEP)
    #print("startCEPlist\t"+str(startCEPlist))

    tfinal=2*measurementtime#t3center+measurementtime#/2

    #make strings from the various input lists
    wlasstr=",".join(map(str,wlaslist))
    wenvstr=",".join(map(str,wenvlist))
    Istr=",".join(map(str,Ilist))
    CEPstr=",".join(map(str,array(startCEPlist)))
    tstartstr=",".join(map(str,tstartlist))
    pulsethetastr=",".join(map(str,pulsethetalist))
    numpulses=len(wlaslist)
    pulsetypestr="2"
    for i in range(numpulses-1):
        pulsetypestr+=", 2"
        
    multiplepulsedict={"#numpulses#":str(numpulses),
                       "#pulsetypelist#":pulsetypestr, "#omega2list#":wlasstr,
                       "#omegalist#":wenvstr, "#intensitylist#":Istr,
                       "#CEPlist#":CEPstr, "#pulsestartlist#":tstartstr,
                       "#pulsethetalist#":pulsethetastr,
                       '#timestep#':str(timestep),
                       "#measurementtimestep#":str(measurementtimestep),
                       "#tfinal#":str(tfinal)}
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

def dirscratchpath(username,symlinkdirpath,dirstr):
    userscratchpath=symlinkdirpath+username+"/"
    dirpath=os.getcwd()+"/"+dirstr
    return userscratchpath+dirpath.split(username+"/")[-1]

def symlinktoscratch(username,symlinkdirpath,dirstr):
    userscratchpath=symlinkdirpath+username+"/"
    symlinkrelpath=(os.getcwd()+"/"+dirstr).split(username+"/")[-1]
    scratchpath=userscratchpath+symlinkrelpath
    #print("making scratch path to directory\t"+scratchpath)
    #make a WALKS directory & symlink to it
    #remove scratch directory if it already exists
    #subprocess.call(["rm","-r",scratchpath])
    #create a new directory which will be symlinked to
    walkspath=scratchpath+"WALKS/"
    subprocess.call(["mkdir","-p",walkspath])
    #symlink to the created directory
    #tmp=os.symlink(walkspath,dirstr+"WALKS")
    subprocess.call(["ln","--symbolic",walkspath,dirstr+"WALKS"])

def symlinktoscratch_fullcalculation(username,symlinkdirpath,dirstr):
    userscratchpath=symlinkdirpath+username+"/"
    symlinkrelpath=(os.getcwd()+"/"+dirstr).split(username+"/")[-1]
    scratchpath=userscratchpath+symlinkrelpath
    subprocess.call(["rm","-r",scratchpath])
    subprocess.call(["mkdir","-p",scratchpath])
    subprocess.call(["ln","--symbolic",scratchpath,dirstr])

def streakingcalculation(resultdir, calcindx, txuvoffsetlist, CEPlist,
                         wenvlist,  wlaslist, Ilist, pulsethetalist, timestep,
                         tmeasure, measurementtimestep):
    dirstr=resultdir+str(calcindx)+"/"
    subprocess.call(["mkdir",dirstr])

    #make a symbolic link to a scratch directory
    #symlinktoscratch(username,symlinkdirpath,dirstr)
    

    masterscriptfile="fullcalculation.sh"
    CEPindx=mod(calcindx,nIRCEP*nXUVCEP)


    txuvindx=int(floor(calcindx/(nIRCEP*nXUVCEP)))
    deltatlist=[txuvoffsetlist[txuvindx]]
    tstartlist=maketstartlist(deltatlist,wenvlist)#maketstartlist_streaking(deltatlist,wenvlist)
    initialrelaxation(masterscriptfile, dirstr)
    multiplepulses(masterscriptfile, dirstr, 1, wlaslist, wenvlist, Ilist,
                   CEPlist[CEPindx], tstartlist, pulsethetalist, timestep,
                   measurementtimestep,  tmeasure) 

#def fullcalculation(resultdir,calcindx,deltat1,deltat2,deltat3,CEP1,CEP2,CEP3,w,w2,I1,I2,I3,Imeasure,timestep,tmeasure,measurementtimestep):
def fullcalculation(resultdir,calcindx,deltatlist,CEPlist,wenvlist,wlaslist,\
                    Ilist,timestep,tmeasure,measurementtimestep,pulsethetalist):
    dirstr=resultdir+str(calcindx)+"/"
    subprocess.call(["mkdir",dirstr])
    masterscriptfile="fullcalculation.sh"

    [I1,I2]=Ilist
    [CEP1,CEP2]=CEPlist
    [wenv1,wenv2]=wenvlist
    [wlas1,wlas2]=wlaslist
    tstartlist=maketstartlist(deltatlist,wenvlist)

    #call subroutines to setup each step of the calculation
    initialrelaxation(masterscriptfile,dirstr)
    multiplepulses(masterscriptfile, dirstr, 1, wlaslist, wenvlist, Ilist,
                   CEPlist, tstartlist, pulsethetalist, timestep,
                   measurementtimestep, tmeasure)
#    excitationpulse(masterscriptfile,dirstr,1,wenv1,wlas1,I1,CEP1,timestep)
#    interpulseinterval(masterscriptfile,dirstr,2,deltat1,timestep)
#    excitationpulse(masterscriptfile,dirstr,3,wenv2,wlas2,I2,CEP2,timestep)
#    interpulseinterval(masterscriptfile,dirstr,4,deltat2,timestep)
#    excitationpulse(masterscriptfile,dirstr,5,wenv3,wlas3,I3,CEP3,timestep)
#    interpulseinterval(masterscriptfile,dirstr,6,deltat3,timestep)
#    #measurementpulse(masterscriptfile,dirstr,7,w,w2,Imeasure,0.,timestep,measurementtimestep)
#    measurmentinterval(masterscriptfile,dirstr,2,tmeasure,timestep,measurementtimestep)

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

def printparameterarray_streaking(paramarray,filename):
    f=open(filename,'w')
    f.write("#indx\ttxuvoffset\tXUVCEP\tIRCEP\n")
    (n,m)=shape(paramarray)
    for i in range(n):
        linestr="\t".join(map(str,paramarray[i,:]))
        f.write(linestr+"\n")
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

def wenvadjust(wenv,timestep):
    #adjust wenv so that pulse takes an integral number of timesteps
    duration=pi/wenv
    durationsteps=ceil(duration/timestep)
    durationsteps=durationsteps+mod(durationsteps,2)#force even number of steps
    #print("durationsteps\t"+str(durationsteps))
    retwenv=(pi/(timestep*durationsteps))
    #print("wenv,retwenv\t"+str(wenv)+"\t"+str(retwenv))
    return retwenv


def wenvlistadjust(wenvlist,timestep):
#adjust envelope frequencies so that there are an integral number of
#timesteps in the envelope period
    retlist=copy(wenvlist)
    for i in range(len(retlist)):
        retlist[i]=wenvadjust(retlist[i],timestep)
    return retlist


####################################################################

#A multimensional calculation will involve a large number of parameter
#choices.  First, set up an array containing the full set of parameter
#choices, then iterate through this array for individual calculation
#runs.

#first, make a directory to hold all of the calculation results
symlinkdirpath="/global/lr3/"
username="zwalters"
evstr=sys.argv[-1]
if(len(sys.argv)<2):
    evstr="22"
evfloat=float(evstr)
resultdir="results_"+str(evstr)+"eV/"
#resultdir=sys.argv[-2]#"results_tmp/"
if(resultdir[-1]!="/"):
    resultdir=resultdir+"/"

#symlinktoscratch_fullcalculation(username,symlinkdirpath,resultdir)
subprocess.call(["mkdir",resultdir])
#remove anything previously written to the scratch directory
#subprocess.call(["rm","-r",dirscratchpath(username,symlinkdirpath,resultdir)])

shutil.copy("./optical_streaking_circular_polarization.py",resultdir)
#shutil.copy("./multiprocessingscript.py",resultdir)
#shutil.copy("./mpilaunchscript.py",resultdir)
templatetargetstr=resultdir+"templates"
if(os.path.exists(templatetargetstr)):
    shutil.rmtree(templatetargetstr)
shutil.copytree("./templates",templatetargetstr)
#shutil.copy("./makejobfile.py",resultdir)
#shutil.copy("./taskfarmer.pbs",resultdir)
#shutil.copy("./taskfarmer.test.pbs",resultdir)

#read grid parameters
tmpfile=open("templates/gridfile.txt")
gridstring="".join(tmpfile.readlines())
tmpfile.close()
#read atom parameters
tmpfile=open("templates/atomfile.txt")
atomstring="".join(tmpfile.readlines())
tmpfile.close()

timestep=.05#.025#.025#.1#0.05
#wxuv=.0565#16./Hrt#16./Hrt#7.9/Hrt#14.4/Hrt#7.884/Hrt#1s2s^2- 1s 2s^2 2p
#wxuv=wenvadjust(wxuv/8.,timestep)#((2*pi)/(1000/aut))#2./Hrt#((2*pi)/(1000/aut))#frequency of sin^2 envelope
wir=nm_to_eV(800)/Hrt
wxuv=evfloat/Hrt#wir*10
I1=3e-4#1.e-3#4.8e-5#1.6e-4
I2=1.e-6
wlaslist=[wir,wxuv]
ncycle=8
wenvlist=array([pi/(11e3*2/aut),pi/(11e3*2/aut)])#array([pi/(12e3*2/aut), pi/(380*2/aut)])#array([pi/(12e3/aut),pi/(400/aut)])#12 fs streaking pulse, 400 as probe pulse
#wenvlist=array([wir/(2.*ncycle),wir/(2.*ncycle)])#array([wir/8,wir/4.])
wenvlist=wenvlistadjust(wenvlist,timestep)#adjust envelope frequencies so that there are an integral number of timesteps in the envelope period
Ilist=[I1,I2]
pulsethetaarray=[[0,0],[0,pi/2]]
polarizationnamelist=["zz","zx"]


######################
#Setting up information for multiple calculations

dE=.08#desired energy resolution in eV
maxE=40.#20.#desired maximum energy range in eV
nphase=41#48#48#128#number of phase points for IR laser
nCEPphase=3

#txuvoffsetlist=[0.]#time offset of XUV pulse wrt center of IR pulse
deltatstart=0.
deltatstop=0.
ddeltat=.1
txuvoffsetlist=arange(deltatstart,deltatstop+ddeltat,ddeltat)*1000/aut

dtdip,Tdipmax,ndtdip=tparams(dE,maxE)
dtdip=ceil(dtdip/timestep)*timestep#make interpulse delays an integral
                               #number of timesteps
measurementtimestep=dtdip#4.#interval between calculating dipole
tmeasure=Tdipmax#max(T3max,T3max_integration)#256#total interval to propagate when calculating dipole

IRCEParray=arange(0.,2.,(2.)/nphase)
XUVCEParray=arange(0.,2.,(2.)/nCEPphase)#[0.,1.]
CEPlist=[]
for (i,j) in product(range(len(XUVCEParray)),range(len(IRCEParray))):
    CEPlist.append([IRCEParray[j],XUVCEParray[i]])
#for i in range(len(XUVCEParray)):
#    for j in range(len(IRCEParray)):
#        CEPlist.append([IRCEParray[j],XUVCEParray[i]])
#print("CEP list after construction\t"+str(CEPlist))


ntxuv=len(txuvoffsetlist)
nIRCEP=len(IRCEParray)
nXUVCEP=len(XUVCEParray)
ncalcs=ntxuv*nIRCEP*nXUVCEP
paramarray=zeros((ncalcs,4))
#for calcindx in range(ncalcs):
#for (i,j,k) in product(range(ntxuv),range(nXUVCEP),range(nIRCEP)):
#    calcindx=i*(nXUVCEP*nIRCEP)+j*(nIRCEP)+k
#    streakingcalculation(resultdir,calcindx,txuvoffsetlist,CEPlist,wenvlist,wlaslist,Ilist,timestep,tmeasure,measurementtimestep)
#    paramarray[calcindx,:]=[calcindx,txuvoffsetlist[i],XUVCEParray[j],IRCEParray[k]]
#
#printparameterarray_streaking(paramarray,resultdir+"parameterkey.txt")

calcindx=0
for (i,j,k) in product(range(ntxuv),range(nXUVCEP),range(nIRCEP)):
    paramarray[calcindx,:]=[calcindx,txuvoffsetlist[i],\
                            XUVCEParray[j],IRCEParray[k]]
    calcindx+=1
printparameterarray_streaking(paramarray,resultdir+"parameterkey.txt")

subprocess.call(["mkdir",resultdir])
for polindx in range(2):
    tmpresultdir=resultdir+polarizationnamelist[polindx]+"/"
    for calcindx in range(ncalcs):
        subprocess.call(["mkdir", tmpresultdir])
        streakingcalculation(tmpresultdir, calcindx, txuvoffsetlist,  CEPlist,
                             wenvlist, wlaslist, Ilist, pulsethetaarray[polindx],
                             timestep, tmeasure, measurementtimestep) 

