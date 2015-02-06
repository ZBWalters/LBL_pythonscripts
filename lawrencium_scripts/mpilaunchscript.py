import sys
import glob
from subprocess import call, check_call
import re
import os
from multiprocessing import *
from time import sleep
#from mpi4py import *
####################################################

#This script launches many instances of a single processor computation
#using mpirun.  In order to ensure that only one computation is
#assigned to a particular processor, it creates a queue of open
#processors.  After a computation is finished, its associated
#processor is added back to the queue.  Worker management is handled using Python's

def makehostfile_slurm(hostfilename):
    nodelist=os.environ['SLURM_JOB_NODELIST'].split(",")
    outfile=open(hostfilename,'w')
    cpuspernode=int(os.environ['SLURM_TASKS_PER_NODE'].split("(x")[0])
    for node in nodelist:
        outfile.write(node.strip()+" slots="+str(cpuspernode)+"\n")
    outfile.close()

def makehostfile(hostfilename):
    #make a formatted mpi nodefile from PBS_NODEFILE
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

#def makenodequeue(nodefile):
#    #make a node queue from the files produced in makehostfile
#    retqueue=Queue()
#    f=open(nodefile)
#    for line in f:
#        [nodestr,stepstr]=line.split()
#        nstep=int(stepstr.split("=")[-1])
#        for i in range(nstep):
#            retqueue.put(nodestr)
#    return retqueue

def makenodequeue_slurm():
    #make a node queue corresponding to the information contained in
    #$SLURM_JOB_NODELIST and $SLURM_TASKS_PER_NODE
    retqueue=Queue()
    nodelist=os.environ['SLURM_JOB_NODELIST'].split(",")
    cpuspernode=int(os.environ['SLURM_TASKS_PER_NODE'].split("(")[0])
    for node in nodelist:
        for i in range(cpuspernode):
            retqueue.put(node)
    return retqueue

def makenodequeue():
    #make a node queue corresponding to the information contained in
    #$PBS_NODEFILE
    retqueue=Queue()
    nodefile=os.environ['PBS_NODEFILE']
    nf=open(nodefile,'r')
    nodelist=nf.readlines()
    nf.close()
    for i in range(len(nodelist)):
        retqueue.put(nodelist[i])
    return retqueue

def natural_sort(l):
    #sort a list such that numbers appear in alphanumeric order rather than
    #the usual unix ordering
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def processingrun(indx):
    #1)go to the directory str(indx)/ (ie, 0-> 0/)
    #2)get the name of a node which has an unoccupied processor from nodequeue
    #3)write a file containing the name of the node that the process should
    #run on
    #4)run fullcalculation.sh on that node
    #5)after fullcalculation.sh has returned, put the name of the node onto
    #the queue of nodes with available processors

    nodename=nodequeue.get()
    print("nodename\t"+str(indx)+"\t"+nodename+"\t"+str(nodequeue.qsize()))
    currentdir=os.getcwd()
    dirstr=dirlist[indx]
    os.chdir(dirstr)
    #nodefile=os.environ['PBS_NODEFILE']
    #os.system("cat "+nodefile)
    newnodefile=open("nodelist.txt",'w')
    newnodefile.write(nodename)
    newnodefile.close()
    #makehostfile(newnodefile)
    #print("newnodefile\t"+newnodefile)
    command="mpirun -n 1 --host "+nodename+" -report-bindings bash fullcalculation.sh"
    print("command\t"+command)
    #command="mpirun -n 1 -hostfile nodelist.txt -report-bindings bash fullcalculation.sh"#"chmctdh Inp=Input.Inp.step_0 >& Out.States.step_0"
    sts=call("pwd")
    #sts=call("ls")
    #sleep(10)
    #print("command\t"+command)
    sts=call(command,shell=True)
    os.chdir(currentdir)
    #childcom=MPI.COMM_SELF.Spawn(sys.executable,args=command,maxprocs=1)
    nodequeue.put(nodename)
    
def rundirlist(dirlist):
    absnmax=len(dirlist)

    p=Pool(maxtasksperchild=1,processes=nprocs)
    #p.map(processingrun,range(nmin,min(nmax,absnmax)),chunksize=1)
    p.map_async(processingrun,range(absnmax),chunksize=1)
    p.close()
    p.join()

def dirname(zdipdirstr):
    return "/".join(zdipdirstr.split("/")[:-1])+"/"

###################
#Main Program

#the python Multiprocessing module creates a pool of n workers which
#are assigned to do m tasks, indexed by an integer.  As soon as a
#worker is free, it is assigned the next unassigned task.  Here we
#want to have as many workers as available cpus, and we want to make
#sure that the next task is launched on a node with an unoccupied
#processor available.  To do this, we will make a queue which contains
#the names of nodes with currently unoccupied processors.  When a
#worker begins a task, it will pop a node's name off of this queue,
#then launch the process on that node.  In this way, we will never
#assign more tasks to a node than there are cpus available.

#parse PBS_Nodefile to make nodequeue
#nodefile=os.environ['PBS_NODEFILE']
#os.system("cat "+nodefile)
#nodefile="nodelist.txt"
nodequeue=makenodequeue_slurm()#makenodequeue()
nprocs=int(nodequeue.qsize())
print("nprocs\t"+str(nprocs))

#while(not nodequeue.empty()):
#    print("node\t"+str(nodequeue.get()))

#nmin=int(sys.argv[-2])
#nmax=int(sys.argv[-1])
#print("nmin\t"+str(nmin))
#print("nmax\t"+str(nmax))


dirlist=natural_sort(glob.glob("[0-9]*/"))
rundirlist(dirlist)
#zdiplist=glob.glob("[0-9]*/ZD*exp*.Dat")
#zdiplist=map(dirname,zdiplist)
#unexecuteddirlist=set(dirlist)-set(zdiplist)
#rundirlist(unexecuteddirlist)
