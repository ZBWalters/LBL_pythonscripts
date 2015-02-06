####This script is a simple example of an mpi master/slave setup.  The
####master assigns a task from the front of the queue to each slave.
####Whenever a slave reports back, the master gives the next task
####available, until the queue is exhausted.

from mpi4py import MPI
from time import sleep
from random import randint


def printn(n):
    print("Rank\t"+str(rank)+"\tinput\t"+str(n)+"\n")
    sleep(randint(1,3))

def nextitem(inplist):
    if(len(inplist)==0):
        return None
    return inplist.pop(0)

####
WORKTAG=0
DIETAG=1
DONETAG=2

#############
#master process subroutines
def master():
    workqueue=list(range(10))
    size=MPI.COMM_WORLD.Get_size()
    status=MPI.Status()
    #send
    for i in range(1,size):
        nextinput=nextitem(workqueue)
        comm.isend(obj=nextinput,dest=i,tag=WORKTAG)

    #send new inputs to any process which returns a value
    while True:
        nextinput=nextitem(workqueue)
        if not nextinput: break
        retval=comm.recv(obj=None,source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        print("return\t"+str(retval)+"\t"+str(status.Get_source())+"\n")
        print ("nextinput\t"+str(nextinput))
        comm.send(obj=nextinput,dest=status.Get_source(),tag=WORKTAG)
        
    print("master exits loop")
    for i in range(1,size):
        comm.isend(obj=None,dest=i,tag=DIETAG)
        
    print("master killed processes")

def slave():
    executeinp=comm.recv(source=0,tag=MPI.ANY_TAG)
    print("slave\t"+str(rank)+"\treceives input "+str(executeinp))
    while(executeinp!=None):
        printn("executeinp\t"+str(executeinp))
        comm.send(obj=True,dest=0,tag=DONETAG)
        executeinp=comm.recv(source=0,tag=MPI.ANY_TAG)
    print("slave\t"+str(rank)+"\texits loop")

#############################
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print( "rank\t"+str(rank)+"\tsize\t"+str(size))

if(rank==0):
    master()
else:
    slave()
