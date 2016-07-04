import numpy
import scipy
import scipy.special

from clean import *
from synthesis import *
from simulate import *

from matplotlib import pylab

def aaf_ns(a, m, c):
    """

    """
    r=numpy.hypot(*ucs(a))
    return scipy.special.pro_ang1(m,m,c,r)



if 1:
    import os
    chome = os.environ['CROCODILE']
    vlas=numpy.genfromtxt("%s/test/VLA_A_hor_xyz.txt" % chome, delimiter=",")
    vobs=genuv(vlas, numpy.arange(0,numpy.pi,0.1) ,  numpy.pi/4)
    yy=genvis(vobs/5, 0.01, 0.01)

if 1:
    majorcycle(2*0.025, 2*15000, vobs/5 , yy, 0.1, 5, 100, 250000)
    

if 0: # some other testing code bits
    mg=exmid(numpy.fft.fftshift(numpy.fft.fft2(aaf(a, 0, 3))),5)
    ws=numpy.arange( p[:,2].min(), p[:,2].max(), wstep)
    wr=zip(ws[:-1], ws[1:]) + [ (ws[-1], p[:,2].max() ) ]
    yy=genvis(vobs/5, 0.001, 0.001)
    d,p,_=doimg(2*0.025, 2*15000, vobs/5, yy, lambda *x: wslicimg(*x, wstep=250))
    pylab.matshow(p[740:850,740:850]); pylab.colorbar(); pylab.show()
    x=numpy.zeros_like(d)
    x[1050,1050]=1
    xuv=numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(x)))
