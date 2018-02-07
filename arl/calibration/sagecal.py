""" Radio interferometric calibration using the SAGE algorithm.

Based on the paper:
Radio interferometric calibration with SAGE.

S Kazemi, S Yatawatta, S Zaroubi, P Lampropoulos, A G de Bruyn, L V E Koopmans, and J Noordam.

The aim of the new generation of radio synthesis arrays such as LOw Frequency ARray (LOFAR) and Square Kilometre Array (SKA) is to achieve much higher sensitivity, resolution and frequency coverage than what is available now, especially at low frequencies. To accomplish this goal, the accuracy of the calibration techniques used is of considerable importance. Moreover, since these telescopes produce huge amounts of data, speed of convergence of calibration is a major bottleneck. The errors in calibration are due to system noise (sky and instrumental) as well as the estimation errors introduced by the calibration technique itself, which we call 'solver noise'. We define solver noise as the 'distance' between the optimal solution (the true value of the unknowns, uncorrupted by the system noise) and the solution obtained by calibration. We present the Space Alternating Generalized Expectation Maximization (SAGE) calibration technique, which is a modification of the Expectation Maximization algorithm, and compare its performance with the traditional least squares calibration based on the level of solver noise introduced by each technique. For this purpose, we develop statistical methods that use the calibrated solutions to estimate the level of solver noise. The SAGE calibration algorithm yields very promising results in terms of both accuracy and speed of convergence. The comparison approaches that we adopt introduce a new framework for assessing the performance of different calibration schemes.

Monthly Notices of the Royal Astronomical Society, 2011 vol. 414 (2) pp. 1656-1666.

http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2011MNRAS.414.1656K&link_type=EJOURNAL

In this code:

- A single theta vector is taken to be the amplitude of a given point source and the gain (phase only) solutions for
that point source for all solution point
   

- The E step for a specific window is the sum of the window data model and the discrepancy between the observed data and the summed (over all windows) data models.
   
   
- The M step for a specific window is the optimisation of the theta vector given the window data model. This involves fitting a point source (flux only for the moment) and fitting for the gain phases.


To run sagecal, you must provide a visibility dataset and a set of skycomponents. The output will be the model
parameters (source flux and gaintable for all skycomponents), and the residual visibility.

Sagecal should only be used once an initial phase calibration has been obtained using an isoplanatic approximation.
"""

import logging

import numpy

from arl.calibration.solvers import solve_gaintable
from arl.calibration.operations import copy_gaintable, apply_gaintable, \
    create_gaintable_from_blockvisibility, qa_gaintable
from arl.data.data_models import BlockVisibility
from arl.imaging import predict_skycomponent_visibility
from arl.skycomponent.operations import copy_skycomponent
from arl.visibility.coalesce import convert_blockvisibility_to_visibility
from arl.visibility.operations import copy_visibility, sum_visibility

log = logging.getLogger(__name__)


def create_sagecal_thetas(vis: BlockVisibility, comps, **kwargs):
    """Create the thetas
    
    Create the data model for each window, from the visibility and the existing components
    
    :param comps:
    :param gt:
    :return:
    """
    gt = create_gaintable_from_blockvisibility(vis, **kwargs)
    thetas = list()
    for i, sc in enumerate(comps):
        new_sc = copy_skycomponent(sc)
        new_sc.flux[...] = 0.0
        thetas.append((new_sc, copy_gaintable(gt)))
    return thetas


def sagecal_fit_component(evis, theta, gain=0.1, **kwargs):
    """Fit a single component to a visibility i.e. A13

    This is the update to the component part of the window

    Just do the amplitude for now

    :param evis:
    :param theta:
    :param kwargs:
    :return:
    """
    cvis = convert_blockvisibility_to_visibility(evis)
    new_comp = copy_skycomponent(theta[0])
    new_flux, _ = sum_visibility(cvis, new_comp.direction)
    new_comp.flux = gain * new_flux + (1.0 - gain) * new_comp.flux
    return new_comp


def sagecal_fit_gaintable(evis, theta, gain=0.1, niter=3, tol=1e-3, **kwargs):
    """Fit a gaintable to a visibility i.e. A13
    
    This is the update to the gain part of the window

    :param evis:
    :param theta:
    :param kwargs:
    :return:
    """
    previous_gt = copy_gaintable(theta[1])
    gt = copy_gaintable(theta[1])
    model_vis = copy_visibility(evis, zero=True)
    model_vis = predict_skycomponent_visibility(model_vis, theta[0])
    gt = solve_gaintable(evis, model_vis, gt=gt, niter=niter, phase_only=True, gain=0.9, tol=1e-4, **kwargs)
    gt.data['gain'][...] = gain * gt.data['gain'][...] + \
                           (1 - gain) * previous_gt.data['gain'][...]
    gt.data['gain'][...] /= numpy.abs(previous_gt.data['gain'][...])
    return gt


def sagecal_e_step(vis: BlockVisibility, evis_all: BlockVisibility, theta, beta=1.0, **kwargs):
    """Calculates E step in equation A12
    
    This is the data model for this window plus the difference between observed data and summed data models

    :param vis:
    :param theta:
    :param kwargs:
    :return:
    """
    evis = copy_visibility(evis_all)
    tvis = copy_visibility(vis, zero=True)
    tvis = predict_skycomponent_visibility(tvis, theta[0])
    tvis = apply_gaintable(tvis, theta[1])
    evis.data['vis'][...] = tvis.data['vis'][...] + \
                            beta * (vis.data['vis'][...] - evis_all.data['vis'][...])
    return evis


def sagecal_e_all(vis: BlockVisibility, thetas, **kwargs):
    """Calculates E step in equation A12
    
    This is the sum of the data models over all windows

    :param vis:
    :param thetas:
    :param kwargs:
    :return:
    """
    evis = copy_visibility(vis, zero=True)
    tvis = copy_visibility(vis, zero=True)
    for i, theta in enumerate(thetas):
        tvis.data['vis'][...] = 0.0
        tvis = predict_skycomponent_visibility(tvis, theta[0])
        tvis = apply_gaintable(tvis, theta[1])
        evis.data['vis'][...] += tvis.data['vis'][...]
    return evis


def sagecal_m_step(evis: BlockVisibility, theta, gain=0.25, **kwargs):
    """Calculates M step in equation A13
    
    This maximises the likelihood of the theta parameters given the existing model data
    
    :param vis:
    :param theta:
    :param kwargs:
    :return:
    """
    return (sagecal_fit_component(evis, theta, gain=gain, **kwargs),
            sagecal_fit_gaintable(evis, theta, gain=gain, **kwargs))


def sagecal_solve(vis, components, niter=10, tol=1e-8, gain=0.25, callback=None, **kwargs):
    """ Solve
    
    Solve by iterating, performing E step and M step.
    
    :param vis:
    :param components:
    :param gaintables:
    :param kwargs:
    :return: The individual data models and the residual visibility
    """
    thetas = create_sagecal_thetas(vis, components, **kwargs)
    
    for iter in range(niter):
        new_thetas = list()
        evis_all = sagecal_e_all(vis, thetas, **kwargs)
        print("Iteration %d" % (iter))
        for window_index, theta in enumerate(thetas):
            evis = sagecal_e_step(vis, evis_all, theta, gain=gain, **kwargs)
            new_theta = sagecal_m_step(evis, theta, gain=gain, **kwargs)
            new_thetas.append(new_theta)
            
            if callback is not None:
                callback(iter, thetas)
                
            flux = new_theta[0].flux[0, 0]
            qa = qa_gaintable(new_theta[1])
            residual = qa.data['residual']
            rms_phase = qa.data['rms-phase']
            print("\t Window %d, flux %s, residual %.3f, rms phase %.3f" % (window_index, str(flux), residual,
                                                                            rms_phase))
        thetas = new_thetas
    
    residual_vis = copy_visibility(vis)
    final_vis = sagecal_e_all(vis, thetas, **kwargs)
    residual_vis.data['vis'][...] = vis.data['vis'][...] - final_vis.data['vis'][...]
    return thetas, residual_vis

