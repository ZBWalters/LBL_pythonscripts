import glob
import sys
from shutil import copyfile
from math import ceil, floor
from numpy import *

###################
def linecount(filename):
    return sum(1 for line in open(filename))

def shortindxlist(lenlist):
    retlist=[]
    maxval=array(lenlist).max()
    for i in range(len(lenlist)):
        if(lenlist[i]<maxval):
            retlist.append(i)
    return retlist

def uncompletedlist(zdipdirlist):
    lenlist=list(map(linecount, zdipdirlist))
    redo_indices=shortindxlist(lenlist)
    retlist=[]
    for i in range(len(redo_indices)):
        retlist.append(zdipdirlist[redo_indices[i]])
    return retlist


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

def maketaskfile_makeup(taskfilename, dirnamelist):
    indxlist=list(map(lambda x: x.split("/")[-2], dirnamelist))
    #print("indices for makeup jobs\t"+str(indxlist))
    newfile=open(taskfilename,'w+')
    for indx in indxlist:
        newfile.write('bash runjob.sh '+indx+"\n")
    newfile.close


def walltimestr(walltime):
    hrs=int(floor(walltime))
    mins=int(floor(60*(walltime-hrs)))
    return str(hrs)+":"+str(mins).zfill(2)+":00"

def jobfiledictionary(jobname,nnodes,corespernode,walltime):
    dict={"#jobname#":jobname, "#nnodes#":str(nnodes), "#corespernode#":str(corespernode), "#walltime#":walltimestr(walltime)}
    return dict

#def jobfiledictionary(nodenum,walltime,nmin,nmax):
#    
#    dict={"#nodeid#":str(nodenum), "#walltime#":walltimestr(walltime), "#nmin#":str(int(nmin)), "#nmax#":str(int(nmax))}
#    return dict

def dictionaryreplace(text,dictionary):
    newtext=text
    for i,j in dictionary.items():#.iteritems():
        newtext=newtext.replace(i,j)
    return newtext

def makeupruns_jobfile(resultdir, maxcalcspercore=100):
    if(resultdir[-1]!="/"):
        resultdir=resultdir+"/"
    tmpjobname=jobname+"."+resultdir[:-1]
    templatefilename="templates/htjobscript.slurm.template"

    #set of all directories
    dirlist=glob.glob(resultdir+"[0-9]*/")
    #print("\ndirlist\t"+str(dirlist))
    #print("\n")
    #set of directories for which ZDipoleexpect.Dat exists
    zdipdirlist=glob.glob(resultdir+"[0-9]*/Dat/ZDipoleexpect.Dat")
    incompletelist=uncompletedlist(zdipdirlist)
    print("incomplete list\t"+str(incompletelist))
    #print("zdipdirlist\t"+str(zdipdirlist))
    #print("\n")
    #strip zdipdirlist from "Dat" onwards for comparison with dirlist
    zdipdirlist2=list(map(lambda x: "/".join(x.split("/")[:2])+"/", zdipdirlist))
    incompletelist_stripped=list(map(lambda x: "/".join(x.split("/")[:2])+"/",
                                     incompletelist))
    #print("zdipdirlist2\t"+str(zdipdirlist2))
    #print("\n")

    #diffdirlist is the set of directories which appear in dirlist but not zdipdirlist
    diffdirlist=list(set(dirlist)-set(zdipdirlist2))+incompletelist_stripped
    #sort for convenience
    diffdirlist.sort()
    print("diffdirlist\t"+str(diffdirlist))
    #print("\n")

    ncalcs=len(diffdirlist)
    print("ncalcs\t"+str(ncalcs))
    print("nnodes, nprocs\t"+str(nnodes)+"\t"+str(nnodes*corespernode))
    
    calcspernode=(ncalcs/nnodes)
    print("calcs per node\t"+str(calcspernode))
    print("calcs per processor\t"+str(calcspernode*1./corespernode))
    print("hours per run\t"+str(hrsperrun))
    print("safety factor\t"+str(safetyfactor))
    walltime=ncalcs*hrsperrun*safetyfactor/(1.*nnodes*corespernode)
    print("estimated walltime\t"+str(walltime))
    
    jobdict=jobfiledictionary(tmpjobname,nnodes,corespernode,walltime)
    newfilename=resultdir+"htjobscript_makeup.slurm"
    taskfilename=resultdir+"taskfile_makeup.txt"
    print("new filename\t"+newfilename)

    maxtasksperjob=maxcalcspercore*nnodes*corespernode
    tasklistindices=list(range(0,ncalcs,maxtasksperjob))
    tasklistindices.append(ncalcs)
    taskfilenamelist=[]
    for i in range(len(tasklistindices)-1):
        nlo=tasklistindices[i]
        nhi=tasklistindices[i+1]
        taskfilename=resultdir+"taskfile_makeup"+str(i)+".txt"
        taskfilenamelist.append(taskfilename)
        maketaskfile_makeup(taskfilename, diffdirlist)
        taskfiledict={"#taskfilename#":taskfilename.split("/")[-1]}
        jobscriptfilename=resultdir+"htjobscript_makeup"+str(i)+".slurm"
        templatereplace(templatefilename,jobscriptfilename,[jobdict, taskfiledict])
#        taskfilenamelist=[taskfilename]
#        maketaskfile(taskfilename,ncalcs)
    copyfile("templates/runjob.sh",resultdir+"runjob.sh")




def makejobfile(resultdir, maxcalcspercore=100):
    if(resultdir[-1]!="/"):
        resultdir=resultdir+"/"
    tmpjobname=jobname+"."+resultdir[:-1]
    templatefilename="templates/htjobscript.slurm.template"

    masterscriptfile=resultdir+"queuesubmission.sh"
    ms=open(masterscriptfile,'w')

    dirlist=glob.glob(resultdir+"[0-9]*/")
    ncalcs=len(dirlist)
    print("ncalcs\t"+str(ncalcs))
    print("nnodes, nprocs\t"+str(nnodes)+"\t"+str(nnodes*corespernode))
    
    calcspernode=(ncalcs/nnodes)
    print("calcs per node\t"+str(calcspernode))
    print("calcs per processor\t"+str(calcspernode*1./corespernode))
    print("hours per run\t"+str(hrsperrun))
    print("safety factor\t"+str(safetyfactor))
    walltime=ncalcs*hrsperrun*safetyfactor/(1.*nnodes*corespernode)
    print("estimated walltime\t"+str(walltime))
    
    jobdict=jobfiledictionary(tmpjobname,nnodes,corespernode,walltime)
    newfilename=resultdir+"htjobscript.slurm"
    taskfilename=resultdir+"taskfile.txt"
    print("new filename\t"+newfilename)

    maxtasksperjob=maxcalcspercore*nnodes*corespernode
    tasklistindices=list(range(0,ncalcs,maxtasksperjob))
    tasklistindices.append(ncalcs)
    taskfilenamelist=[]
    for i in range(len(tasklistindices)-1):
        nlo=tasklistindices[i]
        nhi=tasklistindices[i+1]
        taskfilename=resultdir+"taskfile"+str(i)+".txt"
        taskfilenamelist.append(taskfilename)
        maketaskfile(taskfilename,nmin=nlo, nmax=nhi)
        taskfiledict={"#taskfilename#":taskfilename.split("/")[-1]}
        jobscriptfilename=resultdir+"htjobscript"+str(i)+".slurm"
        templatereplace(templatefilename,jobscriptfilename,[jobdict, taskfiledict])
#        taskfilenamelist=[taskfilename]
#        maketaskfile(taskfilename,ncalcs)
    copyfile("templates/runjob.sh",resultdir+"runjob.sh")



################################
#Main Program

jobname="MCTDHF"
nnodes=8
corespernode=16
hrsperrun=4.#15/60.#1.
safetyfactor=2.#overestimate time for calculation to avoid running out of time

print("argv\t"+str(sys.argv))
#resultdir=sys.argv[-1]#"results_tmp/"
for resultdir in sys.argv[1:]:
    #copyfile("./createHTjobfiles.slurm.py", resultdir+"createHTjobfiles.slurm.py")
    makeupruns_jobfile(resultdir, maxcalcspercore=2)

#for nodenum in range(nnodes):
#    nmin=nodenum*calcspernode
#    nmax=min((nodenum+1)*calcspernode,ncalcs)
#    walltime=(nmax-nmin)*(hrsperrun/corespernode)*safetyfactor
#    nodedict=jobfiledictionary(nodenum,walltime,nmin,nmax)
#    newfilename=resultdir+"multiprocessingjob."+str(nodenum)+".pbs"
#    print("new filename\t"+newfilename)
#    ms.write("qsub multiprocessingjob."+str(nodenum)+".pbs\n")
#    templatereplace(templatefilename,newfilename,[nodedict])
#ms.close()
    
