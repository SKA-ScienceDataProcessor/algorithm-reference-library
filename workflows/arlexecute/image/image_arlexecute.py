import logging

from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.image.gather_scatter import image_scatter_facets, image_gather_facets

log = logging.getLogger(__name__)

def image_arlexecute_map_workflow(im, imfunction, facets=1, overlap=0, taper=None, **kwargs):
    """Apply a function across an image: scattering to subimages, applying the function, and then gathering
    
    :param im: Image to be processed
    :param imfunction: Function to be applied
    :param facets: See image_scatter_facets
    :param overlap: image_scatter_facets
    :param taper: image_scatter_facets
    :param kwargs: kwargs for imfunction
    :return: output image
    """
    
    facets_list = arlexecute.execute(image_scatter_facets, nout=16)(im, facets=facets, overlap=overlap,
                                                                    taper=taper)
    root_list = [arlexecute.execute(imfunction)(facet, **kwargs) for facet in facets_list]
    gathered = arlexecute.execute(image_gather_facets)(root_list, im, facets=facets, overlap=overlap,
                                                       taper=taper)
    return gathered