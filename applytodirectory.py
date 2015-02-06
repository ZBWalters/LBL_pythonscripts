import os
from subprocess import call
from sys import argv
from glob import glob
##This script applies a python script to every directory matched by a glob

globstring=argv[-1]
scriptname=argv[-2]

print("glob string\t"+globstring)
print("script name\t"+scriptname)
dirlist=glob(globstring)
print("directory list\t"+str(dirlist))
currentdir=os.getcwd()
for dirname in dirlist:
    print("applying "+scriptname+" in directory "+dirname)
    os.chdir(dirname)
    call(["python",scriptname])
    os.chdir(currentdir)

