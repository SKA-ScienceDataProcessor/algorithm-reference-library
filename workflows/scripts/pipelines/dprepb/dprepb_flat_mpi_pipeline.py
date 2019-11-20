# coding: utf-8

# # Pipeline processing using MPI

# Run using mpiexec -n nrank python dprepb_flat_mpi_pipeline.py

import numpy

from data_models.parameters import arl_path

from mpi4py import MPI

from data_models.polarisation import PolarisationFrame
from processing_components.visibility.base import create_visibility_from_ms, create_visibility_from_rows
from processing_components.visibility.operations import append_visibility, convert_visibility_to_stokes
from processing_components.visibility.vis_select import vis_select_uvrange

from processing_components.image.deconvolution import deconvolve_cube, restore_cube
from processing_components.image.operations import export_image_to_fits, qa_image
from processing_components.image.gather_scatter import image_gather_channels
from processing_components.imaging.base import create_image_from_visibility
from processing_components.imaging.base import invert_2d

from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow

import logging

import argparse

def init_logging():
    logging.basicConfig(filename='%s/dprepb-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    
    results_dir = arl_path('test_results')
    dask_dir = arl_path('test_results/dask-work-space')
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("comm %s, rank %d, size %d" % (comm, rank, size))
    
    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and mpi')
    parser.add_argument('--npixel', type=int, default=512, help='Number of pixels per axis')
    parser.add_argument('--context', dest='context', default='2d', help='Context: 2d|timeslice|wstack')
    
    args = parser.parse_args()
    if rank == 0:
        print(args)
    
        log = logging.getLogger()
        logging.info("Starting Imaging pipeline")
    
    nchan = 40
    uvmax = 450.0
    nfreqwin = 2
    centre = 0
    cellsize = 0.0004
    npixel = args.npixel
    psfwidth = (((8.0 / 2.35482004503) / 60.0) * numpy.pi / 180.0) / cellsize
    
    context = args.context
    if context == 'wstack':
        vis_slices = 45
        if rank == 0:
            print('wstack processing')
    elif context == 'timeslice':
        if rank == 0:
            print('timeslice processing')
        vis_slices = 2
    else:
        if rank == 0:
            print('2d processing')
        context = '2d'
        vis_slices = 1
    
    input_vis = [arl_path('data/vis/sim-1.ms'), arl_path('data/vis/sim-2.ms')]
    
    import time
    
    start = time.time()
    
    pol_frame = PolarisationFrame("stokesIQUV")
    
    def load_invert_and_deconvolve(c):
        
        v1 = create_visibility_from_ms(input_vis[0], channum=[c])[0]
        v2 = create_visibility_from_ms(input_vis[1], channum=[c])[0]
        vf = append_visibility(v1, v2)
        vf = convert_visibility_to_stokes(vf)
        vf.configuration.diameter[...] = 35.0
        rows = vis_select_uvrange(vf, 0.0, uvmax=uvmax)
        v = create_visibility_from_rows(vf, rows)
        
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
    
    def do_list(chan_list):
        print("In rank %d: %s" % (rank, str(chan_list)))
        return [load_invert_and_deconvolve(chan) for chan in chan_list]
    
    if rank == 0:
        subchannels = numpy.array_split(range(nchan), size)
    else:
        subchannels = list()
    
    channels = comm.scatter(subchannels, root=0)
    restored_images = do_list(channels)
    restored_list = comm.gather(restored_images)

    if rank ==0:
        print('About to assemble cubes')
        restored_list = [item for sublist in restored_list for item in sublist]
        restored_cube = image_gather_channels(restored_list)
        print("Processing took %.3f s" % (time.time() - start))

        print(qa_image(restored_cube, context='CLEAN restored cube'))
        export_image_to_fits(restored_cube, '%s/dprepb_arlexecute_%s_clean_restored_cube.fits' % (results_dir, context))