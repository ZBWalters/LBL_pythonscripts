import sys

def dictionaryreplace(text,dictionary):
    newtext=text
    for i,j in dictionary.items():#.iteritems():
        newtext=newtext.replace(i,j)
    return newtext

def filereplace(filename,dictionary):
    f=open(filename,'r')
    linearray=f.readlines()
    for i in range(len(linearray)):
        linearray[i]=dictionaryreplace(linearray[i],dictionary)
    f.close()
    f=open(filename,'w')
    for line in linearray:
        f.write(line)
    f.close()

#########################################
arglist=sys.argv
print("arglist\t"+str(arglist))
oldstring=arglist[1]
newstring=arglist[2]
dict1={oldstring:newstring}

if(len(arglist)>3):
    for i in range(3,len(arglist)):
        filereplace(arglist[i],dict1)
