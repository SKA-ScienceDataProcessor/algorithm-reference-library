# coding: utf-8

# # Pipeline processing using Dask

import numpy

from arl.data_models.parameters import arl_path

results_dir = arl_path('test_results')

from arl.data_models import PolarisationFrame
from arl.processing_components import create_visibility_from_ms, create_visibility_from_rows, \
    append_visibility, convert_visibility_to_stokes, vis_select_uvrange, deconvolve_cube, restore_cube, \
    export_image_to_fits, qa_image, image_gather_channels, create_image_from_visibility, invert_2d

from arl.workflows import invert_list_serial_workflow

import logging

import argparse

def init_logging():
    logging.basicConfig(filename='%s/dprepb-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Benchmark pipelines')
    parser.add_argument('--npixel', type=int, default=512, help='Number of pixels per axis')
    parser.add_argument('--context', dest='context', default='2d', help='Context: 2d|timeslice|wstack')
    parser.add_argument('--nchan', type=int, default=40, help='Number of channels to process')

    args = parser.parse_args()
    print(args)
    
    log = logging.getLogger()
    logging.info("Starting Imaging pipeline")
    
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
    
    def load_invert_and_deconvolve(c):
    
        v1 = create_visibility_from_ms(input_vis[0], channum=[c])[0]
        v2 = create_visibility_from_ms(input_vis[1], channum=[c])[0]
        vf = append_visibility(v1, v2)
        vf = convert_visibility_to_stokes(vf)
        vf.configuration.diameter[...] = 35.0
        rows = vis_select_uvrange(vf, 0.0, uvmax=uvmax)
        v = create_visibility_from_rows(vf, rows)
    
        pol_frame = PolarisationFrame("stokesIQUV")

        m = create_image_from_visibility(v, npixel=npixel, cellsize=cellsize,
                                         polarisation_frame=pol_frame)
    
        if context == '2d':
            d, sumwt = invert_2d(v, m, dopsf=False)
            p, sumwt = invert_2d(v, m, dopsf=True)
        else:
            d, sumwt = invert_list_serial_workflow([v], [m], context=context, dopsf=False,
                                               vis_slices=vis_slices)[0]
            p, sumwt = invert_list_serial_workflow([v], [m], context=context, dopsf=True,
                                               vis_slices=vis_slices)[0]
        c, resid = deconvolve_cube(d, p, m, threshold=0.01, fracthresh=0.01, window_shape='quarter',
                                   niter=100, gain=0.1, algorithm='hogbom-complex')
        r = restore_cube(c, p, resid, psfwidth=psfwidth)
        return r
    
    
    print('About assemble cubes and deconvolve each frequency')
    restored_list = [load_invert_and_deconvolve(c) for c in range(nchan)]
    restored_cube = image_gather_channels(restored_list)
    print("Processing took %.3f s" % (time.time() - start))

    print(qa_image(restored_cube, context='CLEAN restored cube'))
    export_image_to_fits(restored_cube, '%s/dprepb_serial_%s_clean_restored_cube.fits' % (results_dir, context))
    