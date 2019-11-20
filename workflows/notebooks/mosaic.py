import os
import sys

sys.path.append(os.path.join('..', '..'))

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

from matplotlib import pylab

import numpy

from matplotlib import pyplot as plt

from data_models.polarisation import PolarisationFrame

from data_models.parameters import arl_path

from processing_components.visibility.base import create_visibility_from_ms
from processing_components.image.operations import show_image
from processing_library.image.operations import copy_image
from processing_components.imaging.primary_beams import create_pb
from processing_components.imaging.base import create_image_from_visibility
from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow

import logging

if __name__ == '__main__':
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    
    # %%
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    pylab.rcParams['image.cmap'] = 'Greys'
    # %%
    vis_list = create_visibility_from_ms(arl_path('data/vis/xcasa.ms'))
    
    for field, vt in enumerate(vis_list):
        uvdist = numpy.sqrt(vt.data['uvw'][:, 0] ** 2 + vt.data['uvw'][:, 1] ** 2)
        plt.clf()
        plt.plot(uvdist, numpy.abs(vt.data['weight']), '.')
        plt.xlabel('uvdist')
        plt.ylabel('Amp Visibility')
        plt.title('Field %d' % (field))
        plt.show()
    
    cellsize = 0.00001
    
    model = create_image_from_visibility(vis_list[0], cellsize=cellsize, npixel=512, nchan=1,
                                         frequency=[0.5*(8435100000.0+8.4851e+09)], channel_bandwidth=[1e8],
                                         imagecentre=vis_list[0].phasecentre,
                                         polarisation_frame=PolarisationFrame('stokesIQUV'))
    mosaic = copy_image(model)
    mosaicsens = copy_image(model)
    work = copy_image(model)
    
    for vt in vis_list:
        channel_model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512, nchan=1,
                                             imagecentre=vis_list[0].phasecentre,
                                             polarisation_frame=PolarisationFrame('stokesIQUV'))
    
        beam = create_pb(channel_model, telescope='VLA', pointingcentre=vt.phasecentre, use_local=False)
        beam.data /= numpy.max(beam.data)
        dirty, sumwt = invert_list_serial_workflow([vt], [channel_model])[0]
        print(sumwt)
        mosaic.data += dirty.data * beam.data
        mosaicsens.data += beam.data ** 2
        show_image(dirty)
        plt.show()
    
    show_image(mosaic, cm='Greys', title='Linear mosaic')
    plt.show()
    show_image(mosaicsens, cm='Greys', title='Linear mosaic sensitivity')
    plt.show()
    
    from processing_components.image.operations import export_image_to_fits
    
    export_image_to_fits(mosaic, "mosaics.fits")
    export_image_to_fits(mosaicsens, "mosaicsens.fits")
# %%
