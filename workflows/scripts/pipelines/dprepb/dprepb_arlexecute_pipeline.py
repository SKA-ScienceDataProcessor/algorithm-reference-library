# coding: utf-8

# # Pipeline processing using Dask

import numpy

from data_models.parameters import arl_path

results_dir = arl_path('test_results')
dask_dir = arl_path('test_results/dask-work-space')

from data_models.polarisation import PolarisationFrame
from wrappers.arlexecute.visibility.base import create_visibility_from_ms, create_visibility_from_rows
from wrappers.arlexecute.visibility.operations import append_visibility, convert_visibility_to_stokes
from wrappers.arlexecute.visibility.vis_select import vis_select_uvrange

from wrappers.arlexecute.image.deconvolution import deconvolve_cube, restore_cube
from wrappers.arlexecute.image.operations import export_image_to_fits, qa_image
from wrappers.arlexecute.image.gather_scatter import image_gather_channels
from wrappers.arlexecute.imaging.base import create_image_from_visibility
from wrappers.arlexecute.imaging.base import advise_wide_field

from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

import logging


def init_logging():
    logging.basicConfig(filename='%s/dprepb-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    
    log = logging.getLogger()
    logging.info("Starting Imaging pipeline")
    
    arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=16e10,
                          local_dir=dask_dir)
    print(arlexecute.client)
    arlexecute.run(init_logging)
    
    nchan = 40
    uvmax = 450.0
    nfreqwin = 2
    centre = 0
    cellsize = 0.0001
    npixel = 1024
    # This is about 9 pixels and causes the astropy.convolve function to take forever. Need to do
    # by FFT
    psfwidth = (((8.0 / 2.35482004503) / 60.0) * numpy.pi / 180.0) / cellsize
    psfwidth = 3.0

    context = 'wstack'
    vis_slices = 51

    input_vis = [arl_path('data/vis/sim-1.ms'), arl_path('data/vis/sim-2.ms')]
    
    
    def load_ms(c):
        v1 = create_visibility_from_ms(input_vis[0], channum=[c])[0]
        v2 = create_visibility_from_ms(input_vis[1], channum=[c])[0]
        vf = append_visibility(v1, v2)
        vf = convert_visibility_to_stokes(vf)
        rows = vis_select_uvrange(vf, 0.0, uvmax=uvmax)
        return create_visibility_from_rows(vf, rows)
    
    
    # Load data from previous simulation
    vis_list = [arlexecute.execute(load_ms)(c) for c in range(nchan)]
    
    print('Reading visibilities')
    vis_list = arlexecute.persist(vis_list)
    vis_list = arlexecute.compute(vis_list, sync=True)
    
    # The vis data are on the workers so we run the advice function on the workers
    # without transfering the data back to the host.
    advice_list = [arlexecute.execute(advise_wide_field)(v, guard_band_image=8.0, delA=0.02,
                                                         wprojection_planes=1)
                   for _, v in enumerate(vis_list)]
    advice_list = arlexecute.compute(advice_list, sync=True)
    print(advice_list[0])
    
    pol_frame = PolarisationFrame("stokesIQUV")
    
    model_list = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vis_list]
    
    model_list = arlexecute.persist(model_list)
    
    dirty_list = invert_list_arlexecute_workflow(vis_list, template_model_imagelist=model_list, context=context,
                                                 vis_slices=vis_slices)
    psf_list = invert_list_arlexecute_workflow(vis_list, template_model_imagelist=model_list, context=context,
                                               dopsf=True, vis_slices=vis_slices)

    def deconvolve_and_restore(d, p, m):
        c, resid = deconvolve_cube(d[0], p[0], m, threshold=0.01, fracthresh=0.01, window_shape='quarter',
                                   niter=1, gain=0.1, algorithm='hogbom-complex')
        restored = restore_cube(c, p[0], resid, psfwidth=psfwidth)
        return restored

    log.info('About to deconvolve and restore each frequency')
    restored_list = [arlexecute.execute(deconvolve_and_restore)(dirty_list[c], psf_list[c], model_list[c])
                    for c in range(nchan)]
    result = arlexecute.compute(restored_list, sync=True)
    restored_cube = image_gather_channels(result)
    print(qa_image(restored_cube, context='CLEAN restored cube'))
    export_image_to_fits(restored_cube, '%s/dprepb_arlexecute_clean_restored_cube.fits' % (results_dir))

    try:
        arlexecute.close()
    except:
        pass
