# coding: utf-8

# # Pipeline processing using Dask

import numpy

from data_models.parameters import arl_path

results_dir = arl_path('test_results')
dask_dir = arl_path('test_results/dask-work-space')

from data_models.polarisation import PolarisationFrame
from processing_components.visibility import create_visibility_from_ms, create_visibility_from_rows
from processing_components.visibility import append_visibility, convert_visibility_to_stokes
from processing_components.visibility import vis_select_uvrange

from processing_components import deconvolve_cube, restore_cube
from processing_components.image import export_image_to_fits, qa_image
from processing_components.image import image_gather_channels
from processing_components.imaging import create_image_from_visibility

from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

import logging

import argparse


def init_logging():
    logging.basicConfig(filename='%s/dprepb-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and dask')
    parser.add_argument('--use_dask', type=str, default='True', help='Use Dask?')
    parser.add_argument('--serial_invert', type=str, default='False',
                        help='Use serial invert?')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads per worker')
    parser.add_argument('--memory', dest='memory', default=8, help='Memory per worker (GB)')
    parser.add_argument('--npixel', type=int, default=512, help='Number of pixels per axis')
    parser.add_argument('--context', dest='context', default='2d', help='Context: 2d|timeslice|wstack')
    parser.add_argument('--nchan', type=int, default=40, help='Number of channels to process')
    
    args = parser.parse_args()
    print(args)
    
    log = logging.getLogger()
    logging.info("Starting Imaging pipeline")
    
    use_dask = args.use_dask == 'True'
    arlexecute.set_client(use_dask=use_dask, threads_per_worker=args.threads,
                          memory_limit=args.memory * 1024 * 1024 * 1024,
                          n_workers=args.nworkers,
                          local_dir=dask_dir)
    arlexecute.run(init_logging)
    
    nchan = args.nchan
    uvmax = 450.0
    nfreqwin = 2
    centre = 0
    cellsize = 0.0004
    npixel = args.npixel
    psfwidth = (((8.0 / 2.35482004503) / 60.0) * numpy.pi / 180.0) / cellsize
    
    context = args.context
    if context == 'wstack':
        vis_slices = 45
        print('wstack processing')
    elif context == 'timeslice':
        print('timeslice processing')
        vis_slices = 2
    else:
        print('2d processing')
        context = '2d'
        vis_slices = 1
    
    input_vis = [arl_path('data/vis/sim-1.ms'), arl_path('data/vis/sim-2.ms')]
    import time
    
    start = time.time()
    
    
    def load_ms(c):
        v1 = create_visibility_from_ms(input_vis[0], channum=[c])[0]
        v2 = create_visibility_from_ms(input_vis[1], channum=[c])[0]
        vf = append_visibility(v1, v2)
        vf = convert_visibility_to_stokes(vf)
        vf.configuration.diameter[...] = 35.0
        rows = vis_select_uvrange(vf, 0.0, uvmax=uvmax)
        return create_visibility_from_rows(vf, rows)
    
    
    vis_list = [arlexecute.execute(load_ms)(c) for c in range(nchan)]
    
    print('Reading visibilities')
    vis_list = arlexecute.persist(vis_list)
    
    pol_frame = PolarisationFrame("stokesIQUV")
    
    model_list = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vis_list]
    
    model_list = arlexecute.persist(model_list)
    
    if args.serial_invert == 'True':
        print("Invert is serial")
        dirty_list = [arlexecute.execute(invert_list_serial_workflow)
                      ([vis_list[i]], template_model_imagelist=[model_list[i]],
                       context=context, vis_slices=vis_slices)[0]
                      for i in range(nchan)]
        psf_list = [arlexecute.execute(invert_list_serial_workflow)
                    ([vis_list[i]], template_model_imagelist=[model_list[i]],
                     context=context, dopsf=True,
                     vis_slices=vis_slices)[0]
                    for i in range(nchan)]
    else:
        print("Invert is parallel")
        dirty_list = invert_list_arlexecute_workflow(vis_list, template_model_imagelist=model_list, context=context,
                                                     vis_slices=vis_slices)
        psf_list = invert_list_arlexecute_workflow(vis_list, template_model_imagelist=model_list, context=context,
                                                   dopsf=True, vis_slices=vis_slices)
    
    
    def deconvolve(d, p, m):
        import time
        c, resid = deconvolve_cube(d[0], p[0], m, threshold=0.01, fracthresh=0.01, window_shape='quarter',
                                   niter=100, gain=0.1, algorithm='hogbom-complex')
        r = restore_cube(c, p[0], resid)
        return r
    
    
    print('About assemble cubes and deconvolve each frequency')
    restored_list = [arlexecute.execute(deconvolve)(dirty_list[c], psf_list[c], model_list[c])
                     for c in range(nchan)]
    restored_cube = arlexecute.execute(image_gather_channels, nout=1)(restored_list)
    #    restored_cube.visualize('dprepb_arlexecute_pipeline.svg')
    restored_cube = arlexecute.compute(restored_cube, sync=True)
    
    print("Processing took %.3f s" % (time.time() - start))
    print(qa_image(restored_cube, context='CLEAN restored cube'))
    export_image_to_fits(restored_cube, '%s/dprepb_arlexecute_%s_clean_restored_cube.fits' % (results_dir, context))
    
    try:
        arlexecute.close()
    except:
        pass
