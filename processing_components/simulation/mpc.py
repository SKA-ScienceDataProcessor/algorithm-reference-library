import logging

from data_models.memory_data_models import SkyModel

log = logging.getLogger()

from processing_components.image.operations import copy_image
from processing_components.visibility.operations import copy_visibility
from processing_components.calibration.operations import copy_gaintable


def expand_skymodel_by_skycomponents(sm, **kwargs):
    """ Expand a sky model so that all components are in separate skymodels

    """
    return [SkyModel(components=[comp],
                     image=copy_image(sm.image),
                     gaintable=copy_gaintable(sm.gaintable),
                     mask=copy_image(sm.mask),
                     fixed=sm.fixed) for comp in sm.components]


def sum_visibility_over_partitions(blockvis_list):
    """Sum all the visibility partitions
    
    :param blockvis_list:
    :return: Single visibility
    """
    result = copy_visibility(blockvis_list[0])
    for i, v in enumerate(blockvis_list):
        if i > 0:
            result.data['vis'] += v.data['vis']
    
    return result


