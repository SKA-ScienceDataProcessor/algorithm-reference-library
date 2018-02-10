""" Visibility operations

"""

import numpy
from scipy.optimize import minimize

from arl.util.coordinate_support import skycoord_to_lmn


def fit_visibility(vis, sc, tol=1e-8, niter=10, **kwargs):
    """Fit a model in Skycomponents to a visibility set
    
    :param vis:
    :param sc:
    :param tol:
    :param niter:
    :param kwargs:
    :return:
    """
    
    assert vis.polarisation_frame.type == 'stokesI', "Currently restricted to stokesI"

    sumwt = numpy.sum(vis.weight)
    tp2 = 4.0 * numpy.pi * numpy.pi / sumwt

    amp2 = (vis.vis * numpy.conjugate(vis.vis)).real
    sumu2 = tp2 * numpy.sum(vis.weight * vis.u * vis.u * amp2)
    sumv2 = tp2 * numpy.sum(vis.weight * vis.v * vis.v * amp2)
    sumuv = tp2 * numpy.sum(vis.weight * vis.u * vis.v * amp2)

    def Jonly(params):
        # Params are flux, l, m
        flux = params[0]
        l = params[1]
        m = params[2]
        coords = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])
        phasor = numpy.exp(-2j * numpy.pi * numpy.dot(vis.uvw, coords))
        v = flux * phasor
        res = v[:, numpy.newaxis] - vis.vis
        J = numpy.sum(vis.weight * (res * numpy.conjugate(res)).real)
        return J

    def Jboth(params):
        # Params are flux, l, m
        flux = params[0]
        l = params[1]
        m = params[2]
        coords = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])
        phasor = numpy.exp(-2j * numpy.pi * numpy.dot(vis.uvw, coords))
        v = flux * phasor
        res = v[:, numpy.newaxis] - vis.vis
        norm_res = + 2j * numpy.pi * params[0] * vis.weight * phasor
        J = numpy.sum(vis.weight * (res * numpy.conjugate(res)).real)
        gradJ = numpy.array([- numpy.dot(phasor, vis.weight).real,
                             - numpy.dot(vis.u, norm_res).imag,
                             - numpy.dot(vis.v, norm_res).imag])
        return J, gradJ

    def hessian(params):
        flux = params[0]

        hess = [[flux * sumwt, 0.0, 0.0 ],
                [ 0.0, flux * sumu2, 0.0],
                [0.0, 0.0, flux * sumv2]]
        
        return hess
    
    x0 = numpy.array([sc.flux[0, 0], 0.0, 0.0])
    print("J only:", Jboth(x0)[0])
    print("Grad J:", Jboth(x0)[1])
    print("Hessian:", hessian(x0))
    
    bounds = ((None, None), (-0.1, -0.1), (-0.1, 0.1))
    method = 'CG'
    method = 'Nelder-Mead'
    options = {'maxiter': niter, 'disp': True}
    #res = minimize(objective, x0, method=method, jac=True, hess=hessian, bounds=bounds, options=options)
    res = minimize(Jonly, x0, method=method, options=options)

    return res
