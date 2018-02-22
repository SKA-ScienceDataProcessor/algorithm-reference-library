""" Visibility operations

"""

import numpy
from scipy.optimize import minimize

from arl.util.coordinate_support import lmn_to_skycoord, skycoord_to_lmn


def fit_visibility(vis, sc, tol=1e-6, niter=20, verbose=False, method='trust-exact', **kwargs):
    """Fit a single component to a visibility
    
    Uses the scipy.optimize.minimize function.
    
    :param vis:
    :param sc: Initial component
    :param tol: Tolerance of fit
    :param niter: Number of iterations
    :param verbose:
    :param method: 'CG', 'BFGS', 'Powell', 'trust-ncg', 'trust-exact', 'trust-krylov': default 'trust-exact'
    :param kwargs:
    :return: component, convergence info as a dictionary
    """
    
    assert vis.polarisation_frame.type == 'stokesI', "Currently restricted to stokesI"

    # These derivative have been calculated using sympy. See visibility_fitting_sympy.py
    def J(params):
        # Params are flux, l, m
        S = params[0]
        l = params[1]
        m = params[2]
        u = vis.u[:, numpy.newaxis]
        v = vis.v[:, numpy.newaxis]
        vobs = vis.vis
        p = numpy.exp( -2j * numpy.pi * (u * l + v * m))
        vres = vobs - S * p
        J = numpy.sum(vis.weight * (vres * numpy.conjugate(vres)).real)
        return J


    def Jboth(params):
        # Params are flux, l, m
        S = params[0]
        l = params[1]
        m = params[2]
        u = vis.u[:, numpy.newaxis]
        v = vis.v[:, numpy.newaxis]
        vobs = vis.vis
        p = numpy.exp( -2j * numpy.pi * (u * l + v * m))
        vres = vobs - S * p
        Vrp = vres * numpy.conjugate(p) * vis.weight
        J = numpy.sum(vis.weight * (vres * numpy.conjugate(vres)).real)
        gradJ = numpy.array([- 2.0 * numpy.sum(Vrp.real),
                             + 4.0 * numpy.pi * S * numpy.sum(u * Vrp.imag),
                             + 4.0 * numpy.pi * S * numpy.sum(v * Vrp.imag)])
        return J, gradJ

    def hessian(params):
        S = params[0]
        l = params[1]
        m = params[2]
        
        u = vis.u[:, numpy.newaxis]
        v = vis.v[:, numpy.newaxis]
        w = vis.w[:, numpy.newaxis]
        wt = vis.weight

        vobs = vis.vis
        p = numpy.exp( -2j * numpy.pi * (u * l + v * m))
        vres = vobs - S * p
        Vrp = vres * numpy.conjugate(p)
        
        hess = numpy.zeros([3,3])
        hess[0,0] = 2.0 * numpy.sum(wt)
        
        hess[0,1] = 4.0 * numpy.pi * numpy.sum(wt * u * Vrp.imag)
        hess[0,2] = 4.0 * numpy.pi * numpy.sum(wt * v * Vrp.imag)
        
        hess[1,1] = 8.0 * numpy.pi**2 * S * numpy.sum(wt * u**2  * (S + Vrp.real))
        hess[1,2] = 8.0 * numpy.pi**2 * S * numpy.sum(wt * u * v * (S + Vrp.real))
        hess[2,2] = 8.0 * numpy.pi**2 * S * numpy.sum(wt * v**2  * (S + Vrp.real))
        
        hess[1,0] = hess[0,1]
        hess[2,0] = hess[0,2]
        hess[2,1] = hess[1,2]

        return hess
    
    # Initialize l,m,n to be in the direction of the component as defined in the frame of
    # visibility phasecentre
    l, m, n = skycoord_to_lmn(sc.direction, vis.phasecentre)

    x0 = numpy.array([sc.flux[0, 0], l, m])

    bounds = ((None, None), (-0.1, -0.1), (-0.1, 0.1))
    options = {'maxiter': niter, 'disp': verbose}
    res={}
    import time
    start = time.time()
    if method == 'BFGS' or method == 'CG' or method == 'Powell':
        res = minimize(J, x0, method=method, options=options, tol=tol)
    elif method == 'Nelder-Mead':
        res = minimize(Jboth, x0, method=method, options=options, tol=tol)
    elif method == 'L-BFGS-B':
        res = minimize(Jboth, x0, method=method, jac=True, bounds=bounds, options=options, tol=tol)
    else:
        res = minimize(Jboth, x0, method=method, jac=True, hess=hessian, options=options, tol=tol)
    
    if verbose:
        print("Solution for %s took %.6f seconds" % (method, time.time() - start))
        print("Solution = %s" % str(res.x))
        print(res)
        
    sc.flux[...] = res.x[0]
    lmn = (res.x[1], res.x[2], 0.0)
    sc.direction = lmn_to_skycoord(lmn, vis.phasecentre)

    return sc, res
