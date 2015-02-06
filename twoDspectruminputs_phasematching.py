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

aut=24.2#attoseconds per atomic unit of time

#this script constructs a master bash script and all input files for
#one calculation run as used in a multidimensional spectroscopy
#calculation.  Component calculations will be set up by replacing the
#appropriate variables in template files located in templates/  

#The full calculation will be run in its own directory, so that the
#mctdhf code can be run in parallel without trying to read/write the
#same files

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

    masterscript=open(dirstr+masterscriptfile,'a')
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

def measurementpulse(masterscriptfile,dirstr,stepindx,w,w2,intensity,phaseshift,timestep):
    inpstr=stepstr(stepindx-1)
    outstr=stepstr(stepindx)
    steproot=stepstr(stepindx)

    bashtemplate="templates/Measurementpulse.Bat.template"
    inputtemplate="templates/Input.Inp.Measurementpulse.template"

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
 

def fullcalculation(resultdir,calcindx,deltat1,deltat2,deltat3,CEP1,CEP2,CEP3,w,w2,I1,I2,I3,Imeasure,timestep):
    dirstr=resultdir+str(calcindx)+"/"
    subprocess.call(["mkdir",dirstr])
    masterscriptfile="fullcalculation.sh"

    #call subroutines to setup each step of the calculation
    initialrelaxation(masterscriptfile,dirstr)
    excitationpulse(masterscriptfile,dirstr,1,w,w2,I1,CEP1,timestep)
    interpulseinterval(masterscriptfile,dirstr,2,deltat1,timestep)
    excitationpulse(masterscriptfile,dirstr,3,w,w2,I2,CEP2,timestep)
    interpulseinterval(masterscriptfile,dirstr,4,deltat2,timestep)
    excitationpulse(masterscriptfile,dirstr,5,w,w2,I3,CEP3,timestep)
    interpulseinterval(masterscriptfile,dirstr,6,deltat3,timestep)
    measurementpulse(masterscriptfile,dirstr,7,w,w2,Imeasure,0.,timestep)

    #execute masterscriptfile to actually perform the calculation
    currentdir=os.getcwd()
    #subprocess.call(["cd",dirstr])
    os.chdir(dirstr)
    subprocess.call(["bash",masterscriptfile])
    os.chdir(currentdir)
    #subprocess.call(["cd",currentdir])
    
#############################
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
    deltat1=parameterarray[i][1]
    deltat2=parameterarray[i][2]
    deltat3=dt1#does not vary for 3 pulse setup
    CEP1=parameterarray[i][3]
    CEP2=parameterarray[i][4]
    CEP3=parameterarray[i][5]
    
    fullcalculation(resultdir,i,deltat1,deltat2,deltat3,CEP1,CEP2,CEP3,w,w2,I1,I2,I3,Imeasure,timestep)


#############################


###################################

#def createscripts(n):
#    filelist=glob.glob("2D_spectroscopy/*.template")
#    
#    #n=int(sys.argv[1])
#    suffix="."+str(n)
#
#
#
#    for filename in filelist:
#        relaxfile=open(filename)
#        relaxfilestring=relaxfile.read()
#
#        dict=constructdictionary(n)
#        relaxfilestring=dictionaryreplace(relaxfilestring,dict)
#
#        newfilename=filename[16:-9]+suffix
#        print( newfilename)
#
#        printfile=open(newfilename,'w+')
#        printfile.write(relaxfilestring)
#        #os.chmod(newfilename,stat. S_IXUSR)
#        relaxfile.close()
#        printfile.close()
#    
#    masterscriptfilename="fullcalculation"+suffix+".sh"
#    masterscriptfile=open(masterscriptfilename,'w+')
#    #os.chmod(masterscriptfilename,stat.S_IXUSR)
#    masterscriptfile.write("bash InitialRelaxation.Bat"+suffix+"\n") 
#    masterscriptfile.write("bash Excitationpulse.Bat"+suffix+"\n")
#    masterscriptfile.write("bash Interpulse_Interval.Bat"+suffix+"\n")
#    masterscriptfile.write("bash Excitationpulse2.Bat"+suffix+"\n")
#    masterscriptfile.write("bash Interpulse_Interval2.Bat"+suffix+"\n")
#    masterscriptfile.write("bash Measurementpulse.Bat"+suffix+"\n") 
#    masterscriptfile.close() 
#    return masterscriptfilename
    
#def processingrun(n):
#    masterscriptfilename=createscripts(n)
#    print( "prepared execution script "+masterscriptfilename)
#    return subprocess.check_call("bash "+masterscriptfilename+" 2>&1 | tee output.txt",shell=True)
#
#def cb(n):
#    print( "finished with process",n)
#######################################################
#
##p=Pool()#(maxtasksperchild=1)
##p.map(processingrun, range(1,51))
###for i in range(1,32):
###    p.apply_async(processingrun,(i,),callback=cb(i))
##p.close()
##p.join()
#for n in range(1,2):#(1,43):
#    processingrun(n)
####################################################################
####################################################################

#A multimensional calculation will involve a large number of parameter
#choices.  First, set up an array containing the full set of parameter
#choices, then iterate through this array for individual calculation
#runs.

#first, make a directory to hold all of the calculation results
resultdir="results/"
subprocess.call(["mkdir",resultdir])

shutil.copy("./twoDspectruminputs_phasematching.py",resultdir)

dt1=.1
ndt1=50
deltat1array=dt1*array(range(1,ndt1+1))

#dt2=.1
#ndt2=1
#deltat2array=dt2*array(range(1,ndt2+1))
deltat2array=array([1.])#short delay between first two pulses and third pulse

CEParray=[[0,0,0],[0,0,.5],[.5,0,1],[.5,0,.5],[1,0,0],[1,0,.5],[1.5,0,1.5],[0,0,1.5],[0,0,1],[.5,0,1.5],[1.5,0,1],[1.5,0,.5]]
#these choices of CEPs are taken from 
#Meyer & Engel, Appl. Phys. B v71, 293-297 (2000)

ncalcs=len(deltat1array)*len(deltat2array)*len(CEParray)
parameterarray=zeros((ncalcs,6))
indx=0
for i in range(len(deltat1array)):
    for j in range(len(deltat2array)):
        for k in range(len(CEParray)):
            parameterarray[indx,0]=indx
            parameterarray[indx,1]=deltat1array[i]
            parameterarray[indx,2]=deltat2array[j]
            parameterarray[indx,3]=CEParray[k][0]
            parameterarray[indx,4]=CEParray[k][1]
            parameterarray[indx,5]=CEParray[k][2]
            indx+=1

printparameterarray(parameterarray,resultdir+"parameterkey.txt")
#savetxt(resultdir+"parameterkey.txt",parameterarray,delimiter='\t')



w=(2*pi)/(400/aut)#frequency of sin^2 envelope
w2=.66#1.3#frequency of driving laser
exciteintensity=1.e-1
measureintensity=1.e-3
I1=exciteintensity
I2=exciteintensity
I3=exciteintensity
Imeasure=measureintensity
timestep=0.05

#for i in range(ncalcs):
#    processingrun(i)

###parallel execution of loop
p=Pool()
p.map(processingrun,range(ncalcs))
p.close()
p.join



