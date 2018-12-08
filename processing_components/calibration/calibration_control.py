"""Manages the calibration context. This take a string and returns a dictionary e.g.

{'T': {'shape': 'scalar', 'timeslice': 'auto', 'phase_only': True, 'first_selfcal': 0},
                'G': {'shape': 'vector', 'timeslice': 60.0, 'phase_only': False, 'first_selfcal': 0},
                'P': {'shape': 'matrix', 'timeslice': 1e4, 'phase_only': False, 'first_selfcal': 0},
                'B': {'shape': 'vector', 'timeslice': 1e5, 'phase_only': False, 'first_selfcal': 0},
                'I': {'shape': 'vector', 'timeslice': 1.0, 'phase_only': True, 'first_selfcal': 0}}

"""

import logging

import numpy

from data_models.memory_data_models import Visibility, BlockVisibility

from ..calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable, qa_gaintable
from ..calibration.calibration import solve_gaintable
from ..visibility.coalesce import convert_visibility_to_blockvisibility, convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)


def create_calibration_controls(**kwargs):
    """Contains all the control information for calibration

    The fields are:
        T: Atmospheric phase
        G: Electronic gains
        P: Polarisation
        B: Bandpass
        I: Ionosphere

    Get this dictionary and then adjust parameters as desired
    
    The calibrate function takes a context string e.g. TGB. It then calibrates each of these Jones matrices in turn.

    :param kwargs:
    :return:
    """

    controls = {'T': {'shape': 'scalar', 'timeslice': 'auto', 'phase_only': True, 'first_selfcal': 0},
                'G': {'shape': 'vector', 'timeslice': 60.0, 'phase_only': False, 'first_selfcal': 0},
                'P': {'shape': 'matrix', 'timeslice': 1e4, 'phase_only': False, 'first_selfcal': 0},
                'B': {'shape': 'vector', 'timeslice': 1e5, 'phase_only': False, 'first_selfcal': 0},
                'I': {'shape': 'vector', 'timeslice': 1.0, 'phase_only': True, 'first_selfcal': 0}}

    return controls


def apply_calibration_function(vis, gaintables, calibration_context='T', controls=None, iteration=0, tol=1e-6,
                               **kwargs):
    """ Calibrate using algorithm specified by calibration_context
    
    The context string can denote a sequence of calibrations e.g. TGB with different timescales.

    :param vis:
    :param model_vis:
    :param calibration_context: calibration contexts in order of correction e.g. 'TGB'
    :param control: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to the 'first_selfcal' field.
    :param kwargs:
    :return: Calibrated data_models, dict(gaintables)
    """

    if controls is None:
        controls = create_calibration_controls(**kwargs)

    # Check to see if changes are required
    changes = False
    for c in calibration_context:
        if iteration >= controls[c]['first_selfcal']:
            changes = True

    if changes:
    
        isVis = isinstance(vis, Visibility)
        if isVis:
            avis = convert_visibility_to_blockvisibility(vis)
        else:
            avis = vis

        assert isinstance(avis, BlockVisibility), avis

        for c in calibration_context:
            if iteration >= controls[c]['first_selfcal']:
                avis = apply_gaintable(avis, gaintables[c], inverse=True, timeslice=controls[c]['timeslice'])
            
        if isVis:
            return convert_blockvisibility_to_visibility(avis)
        else:
            return avis
    else:
        return vis


def calibrate_function(vis, model_vis, calibration_context='T', controls=None, iteration=0, tol=1e-8, **kwargs):
    """ Calibrate using algorithm specified by calibration_context

    The context string can denote a sequence of calibrations e.g. TGB with different timescales.

    :param vis:
    :param model_vis:
    :param calibration_context: calibration contexts in order of correction e.g. 'TGB'
    :param controls: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to the 'first_selfcal' field.
    :param kwargs:
    :return: Calibrated data_models, dict(gaintables)
    """
    gaintables = {}
    
    if controls is None:
        controls = create_calibration_controls(**kwargs)
    
    # Check to see if changes are required
    changes = False
    for c in calibration_context:
        if iteration >= controls[c]['first_selfcal']:
            changes = True

    if changes:
    
        isVis = isinstance(vis, Visibility)
        if isVis:
            avis = convert_visibility_to_blockvisibility(vis)
        else:
            avis = vis
    
        isMVis = isinstance(model_vis, Visibility)
        if isMVis:
            amvis = convert_visibility_to_blockvisibility(model_vis)
        else:
            amvis = model_vis

        assert isinstance(avis, BlockVisibility), avis

        for c in calibration_context:
            if iteration >= controls[c]['first_selfcal']:
                changed = True
                gaintables[c] = \
                    create_gaintable_from_blockvisibility(avis, timeslice=controls[c]['timeslice'])
                gaintables[c] = solve_gaintable(avis, amvis,
                                                timeslice=controls[c]['timeslice'],
                                                phase_only=controls[c]['phase_only'],
                                                crosspol=controls[c]['shape'] == 'matrix',
                                                tol=tol)
                log.debug('calibrate_function: Jones matrix %s, iteration %d' % (c, iteration))
                log.debug(qa_gaintable(gaintables[c], context='Jones matrix %s, iteration %d' % (c, iteration)))
                avis = apply_gaintable(avis, gaintables[c], inverse=True, timeslice=controls[c]['timeslice'])
            else:
                log.debug('calibrate_function: Jones matrix %s not solved, iteration %d' % (c, iteration))
    
        if isVis:
            return convert_blockvisibility_to_visibility(avis), gaintables
        else:
            return avis, gaintables
    else:
        return vis, gaintables


def solve_calibrate_function(vis, model_vis, calibration_context='T', controls=None, iteration=0, tol=1e-6, **kwargs):
    """ Calibrate using algorithm specified by calibration_context

    The context string can denote a sequence of calibrations e.g. TGB with different timescales.

    :param vis:
    :param model_vis:
    :param calibration_context: calibration contexts in order of correction e.g. 'TGB'
    :param controls: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to the 'first_selfcal' field.
    :param kwargs:
    :return: Calibrated data_models, dict(gaintables)
    """
    gaintables = {}
    
    if controls is None:
        controls = create_calibration_controls(**kwargs)

    # Check to see if changes are required
    changes = False
    for c in calibration_context:
        if iteration >= controls[c]['first_selfcal']:
            changes = True

    if changes:
    
        isVis = isinstance(vis, Visibility)
        if isVis:
            avis = convert_visibility_to_blockvisibility(vis)
        else:
            avis = vis
    
        isMVis = isinstance(model_vis, Visibility)
        if isMVis:
            amvis = convert_visibility_to_blockvisibility(model_vis)
        else:
            amvis = model_vis
            
        assert isinstance(avis, BlockVisibility), avis
    
        for c in calibration_context:
            if iteration >= controls[c]['first_selfcal']:
                gaintables[c] = \
                    create_gaintable_from_blockvisibility(avis, timeslice=controls[c]['timeslice'])
                gaintables[c] = solve_gaintable(avis, amvis,
                                                timeslice=controls[c]['timeslice'],
                                                phase_only=controls[c]['phase_only'],
                                                crosspol=controls[c]['shape'] == 'matrix',
                                                tol=tol)
                log.debug(qa_gaintable(gaintables[c], context='Jones matrix %s, iteration %d' % (c, iteration)))
            else:
                log.debug('calibrate_function: Jones matrix %s not solved, iteration %d' % (c, iteration))
                
    return gaintables
