""" Visibility operations

"""

import numpy
from scipy.optimize import minimize

from arl.util.coordinate_support import skycoord_to_lmn


def fit_visibility(vis, sc, tol=1e-7, niter=100, verbose=True, method='BFGS', **kwargs):
    """Fit a single component to a visibility
    
    Uses the scipy.optimize.minimize function.
    
    :param vis:
    :param sc: Initial component
    :param tol: Tolerance of fit
    :param niter: Number of iterations
    :param method: 'CG', 'BFGS', 'Powell', 'trust-ncg', 'trust-exact', 'trust-krylov': default 'BFGS'
    :param kwargs:
    :return: component, convergence info as a dictionary
    """
    
    assert vis.polarisation_frame.type == 'stokesI', "Currently restricted to stokesI"

    def J(params):
        # Params are flux, l, m
        S = params[0]
        l = params[1]
        m = params[2]
        u = vis.u[:, numpy.newaxis]
        v = vis.v[:, numpy.newaxis]
        Vobs = vis.vis
        p = numpy.exp( -2j * numpy.pi * (u * l + v * m))
        Vres = Vobs - S * p
        J = numpy.sum(vis.weight * (Vres * numpy.conjugate(Vres)).real)
        return J


    def Jboth(params):
        # Params are flux, l, m
        S = params[0]
        l = params[1]
        m = params[2]
        u = vis.u[:, numpy.newaxis]
        v = vis.v[:, numpy.newaxis]
        Vobs = vis.vis
        p = numpy.exp( -2j * numpy.pi * (u * l + v * m))
        Vres = Vobs - S * p
        Vrp = Vres * numpy.conjugate(p) * vis.weight
        J = numpy.sum(vis.weight * (Vres * numpy.conjugate(Vres)).real)
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

        Vobs = vis.vis
        p = numpy.exp( -2j * numpy.pi * (u * l + v * m))
        Vres = Vobs - S * p
        Vrp = Vres * numpy.conjugate(p)
        
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

    x0 = numpy.array([sc.flux[0, 0], 0.0, 0.0])

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
        
    sc.flux = res.x[0]
    

    return sc, res
