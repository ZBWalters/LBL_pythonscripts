from numpy import *
from scipy.optimize import *

Hrt=27.21


def acc(t,E0,w0):
    return E0*cos(w0*t)

def vel(t,E0,w0,t0,v0):
    return v0+(E0/w0)*sin(w0*t)-(E0/w0)*sin(w0*t0)

def pos(t,E0,w0,t0,v0,x0):
    return x0+v0*(t-t0)-(E0/w0)*sin(w0*t0)*(t-t0)-(E0/pow(w0,2))*cos(w0*t)+(E0/pow(w0,2))*cos(w0*t0)


def kplus(wx,Ip):
    return sqrt(2*(wx-Ip)/Hrt)

def kminus(wx,Ip):
    return -kplus(wx,Ip)

def returntime(t0,E0,w0):
    v0=0.
    x0=0.
    tguess=t0+(2.5*pi)/w0
    #print("tguess "+str(tguess)+str(pos(tguess,E0,w0,t0,v0,x0)))
    x0=0.
    v0=0.
    #return newton(pos,tguess,args=(E0,w0,t0,v0,x0),fprime=lambda t,E0,w0,t0,v0,x0:vel(t,E0,w0,t0,v0))
    return brentq(pos,t0+.1,t0+3*pi/w0,args=(E0,w0,t0,v0,x0))

def posarray(tarray,E0,w0,t0,v0,x0):
    return list(map(lambda t:pos(t,E0,w0,t0,v0,x0),tarray))
