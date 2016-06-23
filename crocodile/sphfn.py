# -*- coding: utf-8 -*-
"""Module to evaluate spheroidal functions.

Ported from AIPS 31DEC15, SPHFN.FOR
"""

import numpy
import math


def sphfn(eta, exponent_id, support, gridding=True):
    """Evaluate approximations to zero-order spheroidal functions.

    This is a port of AIPS SPHFN.FOR.

    Args:
        eta (float): Variable which ranges from 0.0 at the centre of the
                     convolution function to 1.0 at its edge. Also from 0.0
                     at the center of the grid correction function to 1.0 at
                     the edge of the map.
        exponent_id (int): Weighting exponent type. Value in range 0-4.
                           Maps to exponents: 0.0, 0.5, 1.0, 1.5, and 2.0
        support (int): The support width of the function.
                       Allowed values: 4, 5, 6, 7, and 8.
        gridding (bool): Flag determines if the spheroidal function or its FT
                      should be generated.
                      If True, generate the function appropriate for gridding.
                      If False, generate its FT.
                      (default=True)
    Returns (float):
        Value of the spheroidal function (or its FT) for the specified eta.

    """
    # Check function arguments
    if exponent_id not in [0, 1, 2, 3, 4]:
        raise ValueError('Invalid exponent_id parameter')
    if support not in [4, 5, 6, 7, 8]:
        raise ValueError('Invalid im parameter')
    if math.fabs(eta) > 1.0:
        raise ValueError('abs(eta) must be <= 1.0')

    # Data tables.
    alpha = numpy.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype='f8')
    p4 = [
        1.584774e-2, -1.269612e-1, 2.333851e-1, -1.636744e-1, 5.014648e-2,
        3.101855e-2, -1.641253e-1, 2.385500e-1, -1.417069e-1, 3.773226e-2,
        5.007900e-2, -1.971357e-1, 2.363775e-1, -1.215569e-1, 2.853104e-2,
        7.201260e-2, -2.251580e-1, 2.293715e-1, -1.038359e-1, 2.174211e-2,
        9.585932e-2, -2.481381e-1, 2.194469e-1, -8.862132e-2, 1.672243e-2]
    p4 = numpy.array(p4, dtype='f8', order='F').reshape((5, 5), order='F')
    q4 = [
        4.845581e-1, 7.457381e-2, 4.514531e-1, 6.458640e-2, 4.228767e-1,
        5.655715e-2, 3.978515e-1, 4.997164e-2, 3.756999e-1, 4.448800e-2]
    q4 = numpy.array(q4, dtype='f8', order='F').reshape((2, 5), order='F')
    p5 = [
        3.722238e-3, -4.991683e-2, 1.658905e-1, -2.387240e-1, 1.877469e-1,
        -8.159855e-2, 3.051959e-2, 8.182649e-3, -7.325459e-2, 1.945697e-1,
        -2.396387e-1, 1.667832e-1, -6.620786e-2, 2.224041e-2, 1.466325e-2,
        -9.858686e-2, 2.180684e-1, -2.347118e-1, 1.464354e-1, -5.350728e-2,
        1.624782e-2, 2.314317e-2, -1.246383e-1, 2.362036e-1, -2.257366e-1,
        1.275895e-1, -4.317874e-2, 1.193168e-2, 3.346886e-2, -1.503778e-1,
        2.492826e-1, -2.142055e-1, 1.106482e-1, -3.486024e-2, 8.821107e-3]
    p5 = numpy.array(p5, dtype='f8', order='F').reshape((7, 5), order='F')
    q5 = [
        2.418820e-1, 2.291233e-1, 2.177793e-1, 2.075784e-1, 1.983358e-1]
    q5 = numpy.array(q5, dtype='f8')
    p6l = [
        5.613913e-2, -3.019847e-1, 6.256387e-1, -6.324887e-1, 3.303194e-1,
        6.843713e-2, -3.342119e-1, 6.302307e-1, -5.829747e-1, 2.765700e-1,
        8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1,
        9.675562e-2, -3.922489e-1, 6.197133e-1, -4.857470e-1, 1.934013e-1,
        1.124069e-1, -4.172349e-1, 6.069622e-1, -4.405326e-1, 1.618978e-1]
    p6l = numpy.array(p6l, dtype='f8', order='F').reshape((5, 5), order='F')
    q6l = [
        9.077644e-1, 2.535284e-1, 8.626056e-1, 2.291400e-1, 8.212018e-1,
        2.078043e-1, 7.831755e-1, 1.890848e-1, 7.481828e-1, 1.726085e-1]
    q6l = numpy.array(q6l, dtype='f8', order='F').reshape((2, 5), order='F')
    p6u = [
        8.531865e-4, -1.616105e-2, 6.888533e-2, -1.109391e-1, 7.747182e-2,
        2.060760e-3, -2.558954e-2, 8.595213e-2, -1.170228e-1, 7.094106e-2,
        4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2,
        6.887946e-3, -4.994202e-2, 1.168451e-1, -1.207733e-1, 5.744210e-2,
        1.071895e-2, -6.404749e-2, 1.297386e-1, -1.194208e-1, 5.112822e-2]
    p6u = numpy.array(p6u, dtype='f8', order='F').reshape((5, 5), order='F')
    q6u = [
        1.101270e+0, 3.858544e-1, 1.025431e+0, 3.337648e-1, 9.599102e-1,
        2.918724e-1, 9.025276e-1, 2.575336e-1, 8.517470e-1, 2.289667e-1]
    q6u = numpy.array(q6u, dtype='f8', order='F').reshape((2, 5), order='F')
    p7l = [
        2.460495e-2, -1.640964e-1, 4.340110e-1, -5.705516e-1, 4.418614e-1,
        3.070261e-2, -1.879546e-1, 4.565902e-1, -5.544891e-1, 3.892790e-1,
        3.770526e-2, -2.121608e-1, 4.746423e-1, -5.338058e-1, 3.417026e-1,
        4.559398e-2, -2.362670e-1, 4.881998e-1, -5.098448e-1, 2.991635e-1,
        5.432500e-2, -2.598752e-1, 4.974791e-1, -4.837861e-1, 2.614838e-1]
    p7l = numpy.array(p7l, dtype='f8', order='F').reshape((5, 5), order='F')
    q7l = [
        1.124957e+0, 3.784976e-1, 1.075420e+0, 3.466086e-1, 1.029374e+0,
        3.181219e-1, 9.865496e-1, 2.926441e-1, 9.466891e-1, 2.698218e-1]
    q7l = numpy.array(q7l, dtype='f8', order='F').reshape((2, 5), order='F')
    p7u = [
        1.924318e-4, -5.044864e-3, 2.979803e-2, -6.660688e-2, 6.792268e-2,
        5.030909e-4, -8.639332e-3, 4.018472e-2, -7.595456e-2, 6.696215e-2,
        1.059406e-3, -1.343605e-2, 5.135360e-2, -8.386588e-2, 6.484517e-2,
        1.941904e-3, -1.943727e-2, 6.288221e-2, -9.021607e-2, 6.193000e-2,
        3.224785e-3, -2.657664e-2, 7.438627e-2, -9.500554e-2, 5.850884e-2]
    p7u = numpy.array(p7u, dtype='f8', order='F').reshape((5, 5), order='F')
    q7u = [
        1.450730e+0, 6.578685e-1, 1.353872e+0, 5.724332e-1, 1.269924e+0,
        5.032139e-1, 1.196177e+0, 4.460948e-1, 1.130719e+0, 3.982785e-1]
    q7u = numpy.array(q7u, dtype='f8', order='F').reshape((2, 5), order='F')
    p8l = [
        1.378030e-2, -1.097846e-1, 3.625283e-1, -6.522477e-1, 6.684458e-1,
        -4.703556e-1, 1.721632e-2, -1.274981e-1, 3.917226e-1, -6.562264e-1,
        6.305859e-1, -4.067119e-1, 2.121871e-2, -1.461891e-1, 4.185427e-1,
        -6.543539e-1, 5.904660e-1, -3.507098e-1, 2.580565e-2, -1.656048e-1,
        4.426283e-1, -6.473472e-1, 5.494752e-1, -3.018936e-1, 3.098251e-2,
        -1.854823e-1, 4.637398e-1, -6.359482e-1, 5.086794e-1, -2.595588e-1]
    p8l = numpy.array(p8l, dtype='f8', order='F').reshape((6, 5), order='F')
    q8l = [
        1.076975e+0, 3.394154e-1, 1.036132e+0, 3.145673e-1, 9.978025e-1,
        2.920529e-1, 9.617584e-1, 2.715949e-1, 9.278774e-1, 2.530051e-1]
    q8l = numpy.array(q8l, dtype='f8', order='F').reshape((2, 5), order='F')
    p8u = [
        4.290460e-5, -1.508077e-3, 1.233763e-2, -4.091270e-2, 6.547454e-2,
        -5.664203e-2, 1.201008e-4, -2.778372e-3, 1.797999e-2, -5.055048e-2,
        7.125083e-2, -5.469912e-2, 2.698511e-4, -4.628815e-3, 2.470890e-2,
        -6.017759e-2, 7.566434e-2, -5.202678e-2, 5.259595e-4, -7.144198e-3,
        3.238633e-2, -6.946769e-2, 7.873067e-2, -4.889490e-2, 9.255826e-4,
        -1.038126e-2, 4.083176e-2, -7.815954e-2, 8.054087e-2, -4.552077e-2]
    p8u = numpy.array(p8u, dtype='f8', order='F').reshape((6, 5), order='F')
    q8u = [
        1.379457e+0, 5.786953e-1, 1.300303e+0, 5.135748e-1, 1.230436e+0,
        4.593779e-1, 1.168075e+0, 4.135871e-1, 1.111893e+0, 3.744076e-1]
    q8u = numpy.array(q8u, dtype='f8', order='F').reshape((2, 5), order='F')

    # Evaluate function for given support and exponent.
    eta2 = eta ** 2
    j = exponent_id
    if support == 4:
        x = eta2 - 1.0
        psi = x * p4[4, j]
        psi = x * (p4[3, j] + psi)
        psi = x * (p4[2, j] + psi)
        psi = x * (p4[1, j] + psi)
        psi = p4[1, j] + psi
        psi /= (1.0 + x * (q4[0, j] + x * q4[1, j]))
    elif support == 5:
        x = eta2 - 1.0
        psi = (p5[0, j] + x * (p5[1, j] + x * (p5[2, j] + x * (p5[3, j] + \
                                                               x * (p5[4, j] + x * (p5[5, j] + x * p5[6, j])))))) / (
                  1.0 +
                  x * q5[j])
    elif support == 6:
        if math.fabs(eta) > 0.75:
            x = eta2 - 1.0
            psi = (p6u[0, j] + x * (p6u[1, j] + x * (p6u[2, j] +
                                                     x * (p6u[3, j] + x * p6u[4, j])))) / (1.0 + x * (q6u[0, j] +
                                                                                                      x * q6u[1, j]))
        else:
            x = eta2 - 0.5625
            psi = (p6l[0, j] + x * (p6l[1, j] + x * (p6l[2, j] +
                                                     x * (p6l[3, j] + x * p6l[4, j])))) / (1.0 + x * (q6l[0, j] +
                                                                                                      x * q6l[1, j]))
    elif support == 7:
        if math.fabs(eta) > 0.775:
            x = eta2 - 1.0
            psi = (p7u[0, j] + x * (p7u[1, j] + x * (p7u[2, j] +
                                                     x * (p7u[3, j] + x * p7u[4, j])))) / (1.0 + x * (q7u[0, j] +
                                                                                                      x * q7u[1, j]))
        else:
            x = eta2 - 0.600625
            psi = (p7l[0, j] + x * (p7l[1, j] + x * (p7l[2, j] +
                                                     x * (p7l[3, j] + x * p7l[4, j])))) / (1.0 + x * (q7l[0, j] +
                                                                                                      x * q7l[1, j]))
    elif support == 8:
        if math.fabs(eta) > 0.775:
            x = eta2 - 1.0
            psi = (p8u[0, j] + x * (p8u[1, j] + x * (p8u[2, j] +
                                                     x * (p8u[3, j] + x * (p8u[4, j] + x * p8u[5, j]))))) / (1.0 +
                                                                                                             x * (q8u[
                                                                                                                      0, j] + x *
                                                                                                                  q8u[
                                                                                                                      1, j]))
        else:
            x = eta2 - 0.600625
            psi = (p8l[0, j] + x * (p8l[1, j] + x * (p8l[2, j] +
                                                     x * (p8l[3, j] + x * (p8l[4, j] + x * p8l[5, j]))))) / (1.0 +
                                                                                                             x * (q8l[
                                                                                                                      0, j] + x *
                                                                                                                  q8l[
                                                                                                                      1, j]))
    else:
        raise ValueError('invalid support width')
    if not gridding or exponent_id == 0 or eta == 0.0:
        return psi
    if math.fabs(eta) == 1.0:
        return 0.0

    return math.pow((1.0 - eta2), alpha[exponent_id]) * psi
