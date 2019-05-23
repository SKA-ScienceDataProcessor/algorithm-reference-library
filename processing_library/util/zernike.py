#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@file py102-example2-zernike.py
@brief Fitting a surface in Python example for Python 102 lecture
@author Tim van Werkhoven (t.i.m.vanwerkhoven@gmail.com)
@url http://python101.vanwerkhoven.org
@date 20111012

Created by Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl) on 2011-10-12
Copyright (c) 2011 Tim van Werkhoven. All rights reserved.

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

### Libraries

import numpy as N
from scipy.misc import factorial as fac

### Init functions
def zernike_rad(m, n, rho):
	"""
	Calculate the radial component of Zernike polynomial (m, n) 
	given a grid of radial coordinates rho.
	
	>>> zernike_rad(3, 3, 0.333)
	0.036926037000000009
	>>> zernike_rad(1, 3, 0.333)
	-0.55522188900000002
	>>> zernike_rad(3, 5, 0.12345)
	-0.007382104685237683
	"""
	
	if (n < 0 or m < 0 or abs(m) > n):
		raise ValueError
	
	if ((n-m) % 2):
		return rho*0.0
	
	pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
	
	return sum(pre_fac(k) * rho**(n-2.0*k) for k in xrange((n-m)/2+1))

def zernike(m, n, rho, phi):
	"""
	Calculate Zernike polynomial (m, n) given a grid of radial
	coordinates rho and azimuthal coordinates phi.
	
	>>> zernike(3,5, 0.12345, 1.0)
	0.0073082282475042991
	>>> zernike(1, 3, 0.333, 5.0)
	-0.15749545445076085
	"""
	if (m > 0): return zernike_rad(m, n, rho) * N.cos(m * phi)
	if (m < 0): return zernike_rad(-m, n, rho) * N.sin(-m * phi)
	return zernike_rad(0, n, rho)

def zernikel(j, rho, phi):
	"""
	Calculate Zernike polynomial with Noll coordinate j given a grid of radial
	coordinates rho and azimuthal coordinates phi.
	
	>>> zernikel(0, 0.12345, 0.231)
	1.0
	>>> zernikel(1, 0.12345, 0.231)
	0.028264010304937772
	>>> zernikel(6, 0.12345, 0.231)
	0.0012019069816780774
	"""
	n = 0
	while (j > n):
		n += 1
		j -= n
	
	m = -n+2*j
	return zernike(m, n, rho, phi)

