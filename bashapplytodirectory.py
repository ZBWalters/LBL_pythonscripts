import os
from subprocess import call
from sys import argv
from glob import glob
import re
##This script applies a python script to every directory matched by a glob
def natural_sort(l):
    #sort a list such that numbers appear in alphanumeric order rather than the usual unix ordering
    convert=lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
###############################################
globstring=argv[-1]
scriptname=argv[-2]

print("glob string\t"+globstring)
print("argv"+str(argv))
dirlist=natural_sort(glob(globstring))
print("directory list\t"+str(dirlist))
currentdir=os.getcwd()
#for dirname in dirlist:
for i in range(len(dirlist)):
    dirname=dirlist[i]
    print("applying "+scriptname+" in directory "+dirname)
    os.chdir(dirname)
    call(argv[1:-1])
    os.chdir(currentdir)

