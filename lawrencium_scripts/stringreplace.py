import sys
import re

def dictionaryreplace(text,dictionary):
    newtext=text
    sumreplacements=0
    for i,j in dictionary.items():#.iteritems():
        pattern=re.compile(i)
        #newtext=newtext.replace(i,j)
        newtext,nreplace=pattern.subn(j,newtext)
        sumreplacements+=nreplace
    return newtext,sumreplacements

def filereplace(filename,dictionary):
    f=open(filename,'r')
    linearray=f.readlines()
    sumreplacements=0
    for i in range(len(linearray)):
        linearray[i],nreplace=dictionaryreplace(linearray[i],dictionary)
        sumreplacements+=nreplace
    f.close()
    if(sumreplacements>0):
        print(str(sumreplacements)+" replacements in file "+filename)
    f=open(filename,'w')
    for line in linearray:
        f.write(line)
    f.close()

#########################################
arglist=sys.argv
#print("arglist\t"+str(arglist))
oldstring=arglist[1]
newstring=arglist[2]
print("replacing\t"+oldstring+"\twith\t"+newstring)
dict1={oldstring:newstring}

if(len(arglist)>3):
    for i in range(3,len(arglist)):
        filereplace(arglist[i],dict1)
