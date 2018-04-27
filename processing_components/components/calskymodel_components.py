""" Radio interferometric calibration using the SAGE algorithm.

Based on the paper:
Radio interferometric calibration with SAGE.

S Kazemi, S Yatawatta, S Zaroubi, P Lampropoulos, A G de Bruyn, L V E Koopmans, and J Noordam.

The aim of the new generation of radio synthesis arrays such as LOw Frequency ARray (LOFAR) and Square Kilometre Array (SKA) is to achieve much higher sensitivity, resolution and frequency coverage than what is available now, especially at low frequencies. To accomplish this goal, the accuracy of the calibration techniques used is of considerable importance. Moreover, since these telescopes produce huge amounts of data_models, speed of convergence of calibration is a major bottleneck. The errors in calibration are due to system noise (sky and instrumental) as well as the estimation errors introduced by the calibration technique itself, which we call 'solver noise'. We define solver noise as the 'distance' between the optimal solution (the true value of the unknowns, uncorrupted by the system noise) and the solution obtained by calibration. We present the Space Alternating Generalized Expectation Maximization (SAGE) calibration technique, which is a modification of the Expectation Maximization algorithm, and compare its performance with the traditional least squares calibration based on the level of solver noise introduced by each technique. For this purpose, we develop statistical methods that use the calibrated solutions to estimate the level of solver noise. The SAGE calibration algorithm yields very promising results in terms of both accuracy and speed of convergence. The comparison approaches that we adopt introduce a new framework for assessing the performance of different calibration schemes.

Monthly Notices of the Royal Astronomical Society, 2011 vol. 414 (2) pp. 1656-1666.

http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2011MNRAS.414.1656K&link_type=EJOURNAL

In this code:

- A single skymodel vector is taken to be a vector composed of skycomponent, gaintable tuples.
   

- The E step for a specific window is the sum of the window data model and the discrepancy between the observed data and the summed (over all windows) data models.
   
   
- The M step for a specific window is the optimisation of the skymodel vector given the window data model. This involves fitting a skycomponent and fitting for the gain phases.


To run skymodel_cal, you must provide a visibility dataset and a set of skycomponents. The output will be the model
parameters (component and gaintable for all skycomponents), and the residual visibility.

One interpretation is that skymodel_cal decomposes the non-isoplanatic phase calibration into a set of decoupled isoplanatic
problems. It does this in an iterative way:

# Set an initial estimate for each component and the gains associated with that component.

# Predict the data model for all components

# For any component, skymodel, correct the estimated visibility by removing appropriate visibilities for all other skymodel.

# For each skymodel, update the skymodel from the skymodel data model.

skymodel_cal works best if an initial phase calibration has been obtained using an isoplanatic approximation.
"""

import logging

from ..calibration.calskymodel import calskymodel_fit_skymodel, calskymodel_fit_gaintable
from ..calibration.operations import copy_gaintable, apply_gaintable, create_gaintable_from_blockvisibility
from ..component_support.arlexecute import arlexecute
from ..components.imaging_components import sum_predict_results
from ..skymodel.operations import copy_skymodel, predict_skymodel_visibility
from ..visibility.operations import copy_visibility

log = logging.getLogger(__name__)


def initialise_calskymodel_component(vislist, skymodel_list, **kwargs):
    """Create the skymodel

    Create the data model for each window, from the visibility and the existing components

    :param comps:
    :param gt:
    :return:
    """
    
    def create_calskymodel(vis, skymodel):
        gt = create_gaintable_from_blockvisibility(vis, **kwargs)
        return (copy_skymodel(skymodel), copy_gaintable(gt))
    
    return [arlexecute.execute(create_calskymodel, nout=2)(vislist, sm) for sm in skymodel_list]


def calskymodel_expectation_step_component(vislist, evis_all_list, calskymodel_list, **kwargs):
    """Calculates E step in equation A12

    This is the data model for this window plus the difference between observed data and summed data models

    :param evis_all: Sum data models
    :param skymodel: skymodel element being fit
    :param kwargs:
    :return: Data model (i.e. visibility) for this skymodel
    """
    
    def make_e(vis, calskymodel, evis_all):
        # Return the estep for a given skymodel
        evis = copy_visibility(vis)
        tvis = copy_visibility(vis, zero=True)
        tvis = predict_skymodel_visibility(tvis, calskymodel[0])
        tvis = apply_gaintable(tvis, calskymodel[1])
        # E step is the data model for a window plus the difference between the observed data_models
        # and the summed data models or, put another way, its the observed data minus the
        # summed visibility for all other windows
        evis.data['vis'][...] = tvis.data['vis'][...] + vis.data['vis'][...] - evis_all.data['vis'][...]
        return evis
    
    return [arlexecute.execute(make_e)(vislist, csm, evis_all_list) for csm in calskymodel_list]


def calskymodel_expectation_all_component(vislist, calskymodel_list):
    """Calculates E step in equation A12

    This is the sum of the data models over all skymodel, It is a global sync point for calskymodel

    :param vislist: Visibility list
    :param calskymodel_list: list of calskymodel
    :return: Sum of data models (i.e. a single BlockVisibility)
    """
    
    def predict_and_apply(ovis, calskymodel):
        tvis = copy_visibility(ovis, zero=True)
        tvis = predict_skymodel_visibility(tvis, calskymodel[0])
        tvis = apply_gaintable(tvis, calskymodel[1])
        return tvis
    
    evislist = [arlexecute.execute(predict_and_apply)(vislist, csm) for csm in calskymodel_list]
    
    return arlexecute.execute(sum_predict_results, nout=1)(evislist)


def calskymodel_maximisation_step_component(evislist, skymodel_list, **kwargs):
    """Calculates M step in equation A13

    This maximises the likelihood of the skymodel parameters given the existing data model. Note that these are done
    separately rather than jointly.

    :param skymodel:
    :param kwargs:
    :return:
    """
    
    def make_skymodel(ev, skymodel):
        return (calskymodel_fit_skymodel(ev, skymodel, **kwargs),
                calskymodel_fit_gaintable(ev, skymodel, **kwargs))
    
    return [arlexecute.execute(make_skymodel)(evislist[i], skymodel_list[i]) for i, _ in enumerate(evislist)]


def calskymodel_solve_component(vislist, skymodel_list, niter=10, tol=1e-8, gain=0.25, **kwargs):
    """ Solve using calskymodel, dask.delayed wrapper

    Solve by iterating, performing E step and M step.

    :param vis: Initial visibility
    :param components: Initial components to be used
    :param gaintables: Initial gain tables to be used
    :param kwargs:
    :return: A dask graph to calculate the individual data models and the residual visibility
    """
    calskymodel_list = initialise_calskymodel_component(vislist, skymodel_list=skymodel_list, **kwargs)
    
    for iter in range(niter):
        evis_all_list = calskymodel_expectation_all_component(vislist, calskymodel_list)
        evislist = calskymodel_expectation_step_component(vislist, evis_all_list, calskymodel_list, gain=gain, **kwargs)
        new_calskymodel_list = calskymodel_maximisation_step_component(evislist, calskymodel_list, **kwargs)
        calskymodel_list = new_calskymodel_list
    
    final_vislist = calskymodel_expectation_all_component(vislist, calskymodel_list)
    
    def res_vis(vis, final_vis):
        residual_vis = copy_visibility(vis)
        residual_vis.data['vis'][...] = vis.data['vis'][...] - final_vis.data['vis'][...]
        return residual_vis
    
    return arlexecute.execute((calskymodel_list, arlexecute.execute(res_vis)(vislist, final_vislist)))
