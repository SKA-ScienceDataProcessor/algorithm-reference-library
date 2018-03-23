""" Radio interferometric calibration using the SAGE algorithm.

Based on the paper:
Radio interferometric calibration with SAGE.

S Kazemi, S Yatawatta, S Zaroubi, P Lampropoulos, A G de Bruyn, L V E Koopmans, and J Noordam.

The aim of the new generation of radio synthesis arrays such as LOw Frequency ARray (LOFAR) and Square Kilometre Array (SKA) is to achieve much higher sensitivity, resolution and frequency coverage than what is available now, especially at low frequencies. To accomplish this goal, the accuracy of the calibration techniques used is of considerable importance. Moreover, since these telescopes produce huge amounts of data, speed of convergence of calibration is a major bottleneck. The errors in calibration are due to system noise (sky and instrumental) as well as the estimation errors introduced by the calibration technique itself, which we call 'solver noise'. We define solver noise as the 'distance' between the optimal solution (the true value of the unknowns, uncorrupted by the system noise) and the solution obtained by calibration. We present the Space Alternating Generalized Expectation Maximization (SAGE) calibration technique, which is a modification of the Expectation Maximization algorithm, and compare its performance with the traditional least squares calibration based on the level of solver noise introduced by each technique. For this purpose, we develop statistical methods that use the calibrated solutions to estimate the level of solver noise. The SAGE calibration algorithm yields very promising results in terms of both accuracy and speed of convergence. The comparison approaches that we adopt introduce a new framework for assessing the performance of different calibration schemes.

Monthly Notices of the Royal Astronomical Society, 2011 vol. 414 (2) pp. 1656-1666.

http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2011MNRAS.414.1656K&link_type=EJOURNAL

In this code:

- A single ssm vector is taken to be a vector composed of skycomponent, gaintable tuples.

- The E step for a specific window is the sum of the window data model and the discrepancy between the observed data and the summed (over all windows) data models.
   
   
- The M step for a specific window is the optimisation of the ssm vector given the window data model. This involves fitting a skycomponent and fitting for the gain phases.


To run sagecal, you must provide a visibility dataset and a set of skycomponents. The output will be the model
parameters (component and gaintable for all skycomponents), and the residual visibility.

One interpretation is that SageCal decomposes the non-isoplanatic phase calibration into a set of decoupled isoplanatic
problems. It does this in an iterative way:

# Set an initial estimate for each component and the gains associated with that component.

# Predict the data model for all components

# For any component, ssm, correct the estimated visibility by removing appropriate visibilities for all other skymodel.

# For each ssm, update the skymodel from the ssm data model.

Sagecal works best if an initial phase calibration has been obtained using an isoplanatic approximation.
"""

import logging

import numpy

from arl.calibration.solvers import solve_gaintable
from arl.calibration.operations import copy_gaintable, apply_gaintable, \
    create_gaintable_from_blockvisibility, qa_gaintable
from arl.data.data_models import BlockVisibility
from arl.imaging import predict_skycomponent_visibility
from arl.skycomponent.base import copy_skycomponent
from arl.visibility.coalesce import convert_blockvisibility_to_visibility
from arl.visibility.operations import copy_visibility, sum_visibility
from arl.visibility.visibility_fitting import fit_visibility
from arl.data.skymodel import SkyModel

from arl.skymodel.operations import copy_skymodel

log = logging.getLogger(__name__)


def initialise_skymodel_cal_skymodel(vis: BlockVisibility, skymodels, **kwargs):
    """Create the skymodel

    Create the data model for each window, from the visibility and the existing components.
    Note that the components can be lists

    :param vis: BlockVisibility to process
    :param skymodels: List of skyModels
    :param kwargs:
    :return:
    """
    gt = create_gaintable_from_blockvisibility(vis, **kwargs)
    return [(copy_skymodel(sm), copy_gaintable(gt)) for sm in skymodels]


def skymodel_cal_fit_skymodel(vis, calskymodel, gain=0.1, method='fit', **kwargs):
    """Fit a single skymodel to a visibility

    This is the update to the component part of the window.

    :param evis: Expected vis for this ssm
    :param calskymodel: scm element being fit i.e. (skymodel, gaintable) tuple
    :param gain: Gain in step
    :param method: 'fit' or 'sum'
    :param kwargs:
    :return: skycomponent
    """
    cvis = convert_blockvisibility_to_visibility(vis)
    new_comps = list()
    for comp in calskymodel[0].components:
        new_comp = copy_skycomponent(comp)
        if method == 'sum':
            new_flux, _ = sum_visibility(cvis, new_comp.direction)
            new_comp.flux = gain * new_flux + (1.0 - gain) * comp[0].flux
        else:
            new_comp, _ = fit_visibility(cvis, new_comp)
            new_comp.flux = gain * new_comp.flux + (1.0 - gain) * comp.flux
        new_comps.append(new_comp)

    return SkyModel(components=new_comps)

def skymodel_cal_fit_gaintable(evis, calskymodel, gain=0.1, niter=3, tol=1e-3, **kwargs):
    """Fit a gaintable to a visibility
    
    This is the update to the gain part of the window

    :param evis: Expected vis for this ssm
    :param calskymodel: csm element being fit
    :param gain: Gain in step
    :param niter: Number of iterations
    :param kwargs: Gaintable
    """
    previous_gt = copy_gaintable(calskymodel[1])
    gt = copy_gaintable(calskymodel[1])
    model_vis = copy_visibility(evis, zero=True)
    model_vis = predict_skycomponent_visibility(model_vis, calskymodel[0].components)
    gt = solve_gaintable(evis, model_vis, gt=gt, niter=niter, phase_only=True, gain=0.5,
                         tol=1e-4, **kwargs)
    gt.data['gain'][...] = gain * gt.data['gain'][...] + \
                           (1 - gain) * previous_gt.data['gain'][...]
    gt.data['gain'][...] /= numpy.abs(previous_gt.data['gain'][...])
    return gt


def skymodel_cal_e_step(vis: BlockVisibility, evis_all: BlockVisibility, calskymodel, **kwargs):
    """Calculates E step in equation A12

    This is the data model for this window plus the difference between observed data and summed data models

    :param evis_all: Sum data models
    :param csm: csm element being fit
    :param kwargs:
    :return: Data model (i.e. visibility) for this csm
    """
    evis = copy_visibility(evis_all)
    tvis = copy_visibility(vis, zero=True)
    tvis = predict_skycomponent_visibility(tvis, calskymodel[0].components)
    tvis = apply_gaintable(tvis, calskymodel[1])
    evis.data['vis'][...] = tvis.data['vis'][...] + vis.data['vis'][...] - evis_all.data['vis'][...]
    return evis


def skymodel_cal_e_all(vis: BlockVisibility, calskymodels):
    """Calculates E step in equation A12

    This is the sum of the data models over all skymodel

    :param vis: Visibility
    :param csm: List of (skymodel, gaintable) tuples
    :param kwargs:
    :return: Sum of data models (i.e. a visibility)
    """
    evis = copy_visibility(vis, zero=True)
    tvis = copy_visibility(vis, zero=True)
    for csm in calskymodels:
        tvis.data['vis'][...] = 0.0
        tvis = predict_skycomponent_visibility(tvis, csm[0].components)
        tvis = apply_gaintable(tvis, csm[1])
        evis.data['vis'][...] += tvis.data['vis'][...]
    return evis


def skymodel_cal_m_step(evis: BlockVisibility, calskymodel, **kwargs):
    """Calculates M step in equation A13

    This maximises the likelihood of the ssm parameters given the existing data model. Note that these are done
    separately rather than jointly.

    :param ssm:
    :param kwargs:
    :return:
    """
    return (skymodel_cal_fit_skymodel(evis, calskymodel, **kwargs),
            skymodel_cal_fit_gaintable(evis, calskymodel, **kwargs))


def skymodel_cal_solve(vis, skymodels, niter=10, tol=1e-8, gain=0.25, **kwargs):
    """ Solve
    
    Solve by iterating, performing E step and M step.
    
    :param vis: Initial visibility
    :param components: Initial components to be used
    :param gaintables: Initial gain tables to be used
    :param kwargs:
    :return: The individual data models and the residual visibility
    """
    calskymodels = initialise_skymodel_cal_skymodel(vis, skymodels=skymodels, **kwargs)
    
    for iter in range(niter):
        new_calskymodels = list()
        evis_all = skymodel_cal_e_all(vis, calskymodels)
        log.debug("skymodel_cal_solve: Iteration %d" % (iter))
        for window_index, csm in enumerate(calskymodels):
            evis = skymodel_cal_e_step(vis, evis_all, csm, gain=gain, **kwargs)
            new_csm = skymodel_cal_m_step(evis, csm, **kwargs)
            new_calskymodels.append((new_csm[0], new_csm[1]))
            
            flux = new_csm[0].components[0].flux[0, 0]
            qa = qa_gaintable(new_csm[1])
            residual = qa.data['residual']
            rms_phase = qa.data['rms-phase']
            log.debug("skymodel_cal_solve:\t Window %d, flux %s, residual %.3f, rms phase %.3f" % (window_index,
                                                                            str(flux), residual,
                                                                            rms_phase))
            
        calskymodels = [(copy_skymodel(csm[0]), copy_gaintable(csm[1])) for csm in new_calskymodels]
    
    residual_vis = copy_visibility(vis)
    final_vis = skymodel_cal_e_all(vis, calskymodels)
    residual_vis.data['vis'][...] = vis.data['vis'][...] - final_vis.data['vis'][...]
    return calskymodels, residual_vis


