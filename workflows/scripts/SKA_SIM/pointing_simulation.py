# # Simulation of the effect of pointing errors on MID observations


import csv
import socket
import sys
import time

import seqfile

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u

from matplotlib import pyplot as plt

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import Skycomponent, SkyModel
from data_models.data_model_helpers import export_pointingtable_to_hdf5

from wrappers.serial.visibility.base import create_blockvisibility
from wrappers.serial.image.operations import show_image, qa_image
from wrappers.serial.simulation.testing_support import create_named_configuration, simulate_pointingtable
from wrappers.serial.imaging.primary_beams import create_vp, create_pb
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from processing_components.simulation.pointing import create_gaintable_from_pointingtable
from wrappers.arlexecute.visibility.base import copy_visibility

from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility

from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import weight_list_serial_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

from wrappers.arlexecute.execution_support.dask_init import get_dask_Client

import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate pointing errors')
    parser.add_argument('--context', type=str, default='singlesource',
                        help='s3sky or singlesource')
    
    args = parser.parse_args()
    context = args.context
    
    doplot = False
    
    client = get_dask_Client(memory_limit=8 * 1024 * 1024 * 1024)
    arlexecute.set_client(client=client)
    
    nfreqwin = 1
    ntimes = 65
    rmax = 1e5
    diameter = 15.0
    
    frequency = [1.4e9]
    channel_bandwidth = [1e7]
    
    # Trial and error
    HWHM_deg = 1.03 * 180.0 * 3e8 / (numpy.pi * diameter * frequency[0])
    
    print('HWHM beam = %g deg' % HWHM_deg)
    HWHM = HWHM_deg * numpy.pi / 180.0
    
    h2r = numpy.pi / 12.0
    times = numpy.linspace(-6 * h2r, +6 * h2r, ntimes)
    
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    outlier_phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    midcore = create_named_configuration('MID', rmax=rmax)
    
    block_vis = create_blockvisibility(midcore, times, frequency=frequency,
                                       channel_bandwidth=channel_bandwidth, weight=1.0, phasecentre=phasecentre,
                                       polarisation_frame=PolarisationFrame("stokesI"), zerow=True)
    
    vis = convert_blockvisibility_to_visibility(block_vis)
    advice = advise_wide_field(vis, guard_band_image=1.0, delA=0.02)
    
    cellsize = advice['cellsize']
    npixel = 512
    if context == 's3sky':
        pb_npixel = 4096
        pb_cellsize = 4.0 * HWHM / pb_npixel
    else:
        pb_npixel = 4096
        pb_cellsize = HWHM / pb_npixel

    plt.clf()
    plt.plot(vis.u, vis.v, '.')
    plt.show()
    
    model = create_image_from_visibility(block_vis, npixel=npixel, frequency=frequency,
                                         nchan=nfreqwin, cellsize=cellsize, phasecentre=phasecentre)
    
    vis = weight_list_serial_workflow([vis], [model])[0]
    
    if context == 'singlesource':
        # Put a single point source at the phasecentre
        original_components = [Skycomponent(flux=[[1.0]], direction=phasecentre,
                                            frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesI'))]
        
        offset = [180.0 * pb_cellsize * pb_npixel / (2.0 * numpy.pi), 0.0]
        HWHM = HWHM_deg * numpy.pi / 180.0
        # The primary beam is offset to approximately the halfpower point
        pb_direction = SkyCoord(ra=(+15.0 + offset[0] / numpy.cos(-45.0 * numpy.pi / 180.0)) * u.deg,
                                dec=(-45.0 + offset[1]) * u.deg, frame='icrs', equinox='J2000')
    
    else:
        # Make a skymodel from S3
        from wrappers.serial.simulation.testing_support import create_test_skycomponents_from_s3
        
        original_components = create_test_skycomponents_from_s3(flux_limit=0.3,
                                                                phasecentre=phasecentre,
                                                                polarisation_frame=PolarisationFrame("stokesI"),
                                                                frequency=numpy.array(frequency),
                                                                radius=pb_cellsize * pb_npixel)
        HWHM = HWHM_deg * numpy.pi / 180.0
        # Primary beam points to the phasecentre
        pb_direction = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    
    # ### Calculate the voltage patterns with and without pointing errors
    vp = create_image_from_visibility(block_vis, npixel=pb_npixel, frequency=frequency,
                                      nchan=nfreqwin, cellsize=pb_cellsize, phasecentre=phasecentre,
                                      override_cellsize=False)
    
    pb = create_pb(vp, 'MID', pointingcentre=pb_direction)
    show_image(pb)
    plt.show()
    print(pb.data[0, 0, pb_npixel // 2, pb_npixel // 2])
    print(pb.data[0, 0, 0, pb_npixel // 2])
    print(pb.data[0, 0, :, 0])
    
    vp = create_vp(vp, 'MID', pointingcentre=pb_direction)
    pt = create_pointingtable_from_blockvisibility(block_vis, vp)
    
    no_error_pt = simulate_pointingtable(pt, 0.0, 0.0, seed=18051955)
    export_pointingtable_to_hdf5(no_error_pt, 'pointingsim_%s_noerror_pointingtable.hdf5' % context)
    no_error_gt = create_gaintable_from_pointingtable(block_vis, original_components, no_error_pt, vp)
    
    no_error_sm = [SkyModel(components=[original_components[i]], gaintable=no_error_gt[i])
                   for i, _ in enumerate(original_components)]
    
    no_error_vis = copy_visibility(vis)
    no_error_vis = predict_skymodel_list_arlexecute_workflow(no_error_vis, no_error_sm, context='2d',
                                                             docal=True)
    no_error_vis = arlexecute.compute(no_error_vis, sync=True)[0]
    
    static = 0.0
    dynamic = 1.0
    fwhm = 1.0
    pes = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    results = []
    
    filename = seqfile.findNextFile(prefix='pointingsimulation', suffix='.csv')
    print('Saving results to %s' % filename)
    
    epoch = time.strftime("%Y-%m-%d %H:%M:%S")
    
    for pe in pes:
        
        result = dict()
        result['context'] = context
        result['nb_name'] = sys.argv[0]
        result['hostname'] = socket.gethostname()
        result['epoch'] = epoch
        result['npixel'] = npixel
        result['pb_npixel'] = pb_npixel
        
        a2r = numpy.pi / (3600.0 * 180.0)
        static_pointing_error = static * pe
        pointing_error = dynamic * pe
        
        result['static_pointing_error'] = static_pointing_error
        result['dynamic_pointing_error'] = pointing_error
        
        error_pt = simulate_pointingtable(pt, pointing_error=pointing_error * a2r,
                                          static_pointing_error=static_pointing_error * a2r, seed=18051955)
        export_pointingtable_to_hdf5(error_pt,
                                     'pointingsim_%s_error_%.0farcsec_pointingtable.hdf5' % (context, pe))
        
        error_gt = create_gaintable_from_pointingtable(block_vis, original_components, error_pt, vp)
        
        error_sm = [SkyModel(components=[original_components[i]], gaintable=error_gt[i])
                    for i, _ in enumerate(original_components)]
        
        error_vis = copy_visibility(vis)
        error_vis = predict_skymodel_list_arlexecute_workflow(error_vis, error_sm, context='2d',
                                                              docal=True)
        error_vis = arlexecute.compute(error_vis, sync=True)[0]
        
        error_vis.data['vis'] -= no_error_vis.data['vis']
        
        dirty = invert_list_arlexecute_workflow([error_vis], [model], '2d')
        dirty, sumwt = arlexecute.compute(dirty, sync=True)[0]
        show_image(dirty, cm='gray_r', title='Residual image on-source')
        plt.show()
        qa = qa_image(dirty)
        for field in ['maxabs', 'rms', 'medianabs']:
            result["onsource_" + field] = qa.data[field]
        
        outlier_model = create_image_from_visibility(error_vis, npixel=npixel, frequency=frequency,
                                                     nchan=nfreqwin, cellsize=cellsize,
                                                     phasecentre=outlier_phasecentre)
        outlier_dirty = invert_list_arlexecute_workflow([error_vis], [outlier_model], '2d')
        outlier_dirty, outlier_sumwt = arlexecute.compute(outlier_dirty, sync=True)[0]
        show_image(outlier_dirty, cm='gray_r', title='Outlier residual image (dec -35deg)')
        plt.show()
        qa = qa_image(outlier_dirty)
        for field in ['maxabs', 'rms', 'medianabs']:
            result["outlier_" + field] = qa.data[field]
        
        results.append(result)
    
    import pprint
    
    pp = pprint.PrettyPrinter()
    pp.pprint(results)
    
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys(), delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        csvfile.close()

    plt.clf()
    colors = ['r', 'b', 'g']
    for ifield, field in enumerate(['onsource_maxabs', 'onsource_rms', 'onsource_medianabs']):
        plt.loglog(pes, [result[field] for result in results], '-', label=field, color=colors[ifield])
    for ifield, field in enumerate(['outlier_maxabs', 'outlier_rms', 'outlier_medianabs']):
        plt.loglog(pes, [result[field] for result in results], '--', label=field, color=colors[ifield])
    plt.xlabel('Pointing error (arcsec)')
    plt.ylabel('Error (Jy)')
    plt.title('Error for S3SEX sky at %g Hz, %d times, full array: dynamic %g, static %g' %
              (frequency[0], ntimes, dynamic, static))
    plt.legend()
    plotfile = seqfile.findNextFile(prefix='pointingsimulation', suffix='.jpg')
    print('Saving plot to %s' % plotfile)

    plt.savefig(plotfile)
    plt.show()
    
