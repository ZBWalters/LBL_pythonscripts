from numpy import *

def Atod(A,w):
#w in wavenumbers
    c=137
    Ep=1/(4*pi)
    tmpA=A*2.41e-17
    tmpw=w*4.556e-6
    retd=sqrt(3*pi*tmpA*Ep*pow(c,3)/pow(tmpw,3))
    return retd

def nm_to_invcm(nm):
    #return 1e7/nm
#return 2*pi/lambda, where lambda is in units of centimeters
    return 1./(nm*1e-7)#2*pi/(nm*1e-7)

def Atod_nm(A,nm):
    return Atod(A,nm_to_invcm(nm))
