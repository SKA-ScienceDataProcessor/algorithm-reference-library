Wide-field imaging demonstration
================================

This script makes a fake data set, fills it with a number of point
components, and then images it using a variety of algorithms. See
imaging-fits for a similar notebook that checks for errors in the
recovered properties of the images.

The measurement equation for a wide field of view interferometer is:

.. math:: V(u,v,w) =\int \frac{I(l,m)}{\sqrt{1-l^2-m^2}} e^{-2 \pi j (ul+um + w(\sqrt{1-l^2-m^2}-1))} dl dm

We will show various algorithms for computing approximations to this
integral. Calculation of the visibility V from the sky brightness I is
called predict, and the inverese is called invert.

.. code:: ipython3

    %matplotlib inline
    
    import os
    import sys
    
    sys.path.append(os.path.join('..', '..'))
    
    from arl.data_models.parameters import arl_path
    
    results_dir = arl_path('test_results')
    
    from matplotlib import pylab
    
    import numpy
    
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.wcs.utils import pixel_to_skycoord
    
    from matplotlib import pyplot as plt
    
    from arl.data_models.polarisation import PolarisationFrame
    
    from arl.processing_library.image import create_w_term_like
    
    # Use serial wrappers by default
    from arl.processing_components import  create_visibility, create_visibility_from_rows, create_skycomponent,\
        show_image, export_image_to_fits, create_named_configuration, create_image_from_visibility, \
        predict_skycomponent_visibility, advise_wide_field, vis_timeslice_iter, weight_visibility, vis_timeslices, \
        create_awterm_convolutionfunction
    
    # Use arlexecute for imaging
    from arl.wrappers.arlexecute.execution_support.arlexecute import arlexecute
    
    from arl.workflows import invert_list_arlexecute_workflow
    
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    mpl_logger = logging.getLogger("matplotlib") 
    mpl_logger.setLevel(logging.WARNING) 
    
    doplot = True


.. code:: ipython3

    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    pylab.rcParams['image.cmap'] = 'rainbow'

Construct the SKA1-LOW core configuration

.. code:: ipython3

    lowcore = create_named_configuration('LOWBD2-CORE')


.. parsed-literal::

    create_named_configuration: LOWBD2-CORE
    	(<Quantity -2565018.31203579 m>, <Quantity 5085711.90373391 m>, <Quantity -2861033.10788063 m>)
    	GeodeticLocation(lon=<Longitude 116.76444824 deg>, lat=<Latitude -26.82472208 deg>, height=<Quantity 300. m>)
    create_configuration_from_file: 166 antennas/stations


We create the visibility.

This just makes the uvw, time, antenna1, antenna2, weight columns in a
table

.. code:: ipython3

    times = numpy.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) * (numpy.pi / 12.0)
    frequency = numpy.array([1e8])
    channel_bandwidth = numpy.array([1e7])
    
    
    reffrequency = numpy.max(frequency)
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                           weight=1.0, phasecentre=phasecentre, polarisation_frame=PolarisationFrame("stokesI"))


.. parsed-literal::

    create_visibility: 95865 rows, 0.010 GB
    create_visibility: flagged 0/95865 visibilities below elevation limit 0.261799 (rad)


Advise on wide field parameters. This returns a dictionary with all the
input and calculated variables.

.. code:: ipython3

    advice = advise_wide_field(vt, wprojection_planes=1)


.. parsed-literal::

    advise_wide_field: Maximum wavelength 2.998 (meters)
    advise_wide_field: Minimum wavelength 2.998 (meters)
    advise_wide_field: Maximum baseline 262.6 (wavelengths)
    advise_wide_field: Maximum w 169.4 (wavelengths)
    advise_wide_field: Station/dish diameter 35.0 (meters)
    advise_wide_field: Primary beam 0.0857 (rad) 4.91 (deg) 1.77e+04 (asec)
    advise_wide_field: Image field of view 0.514 (rad) 29.4 (deg) 1.06e+05 (asec)
    advise_wide_field: Synthesized beam 0.00381 (rad) 0.218 (deg) 785 (asec)
    advise_wide_field: Cellsize 0.00127 (rad) 0.0727 (deg) 262 (asec)
    advice_wide_field: Npixels per side = 405
    advice_wide_field: Npixels (power of 2) per side = 512
    advice_wide_field: Npixels (power of 2, 3) per side = 512
    advice_wide_field: Npixels (power of 2, 3, 4, 5) per side = 405
    advice_wide_field: W sampling for full image = 0.2 (wavelengths)
    advice_wide_field: W sampling for primary beam = 8.7 (wavelengths)
    advice_wide_field: Time sampling for full image = 25.2 (s)
    advice_wide_field: Time sampling for primary beam = 908.6 (s)
    advice_wide_field: Frequency sampling for full image = 29212.6 (Hz)
    advice_wide_field: Frequency sampling for primary beam = 1051653.8 (Hz)
    advice_wide_field: Number of planes in w stack 39 (primary beam)
    advice_wide_field: Number of planes in w projection 39 (primary beam)
    advice_wide_field: W support = 6 (pixels) (primary beam)


Plot the synthesized UV coverage.

.. code:: ipython3

    if doplot:
        plt.clf()
        plt.plot(vt.data['uvw'][:, 0], vt.data['uvw'][:, 1], '.', color='b')
        plt.plot(-vt.data['uvw'][:, 0], -vt.data['uvw'][:, 1], '.', color='r')
        plt.xlabel('U (wavelengths)')
        plt.ylabel('V (wavelengths)')
        plt.show()
        
        plt.clf()
        plt.plot(vt.data['uvw'][:, 0], vt.data['uvw'][:, 2], '.', color='b')
        plt.xlabel('U (wavelengths)')
        plt.ylabel('W (wavelengths)')
        plt.show()
    
        plt.clf()
        plt.plot(vt.data['time'][vt.u>0.0], vt.data['uvw'][:, 2][vt.u>0.0], '.', color='b')
        plt.plot(vt.data['time'][vt.u<=0.0], vt.data['uvw'][:, 2][vt.u<=0.0], '.', color='r')
        plt.xlabel('U (wavelengths)')
        plt.ylabel('W (wavelengths)')
        plt.show()
    
        plt.clf()
        n, bins, patches = plt.hist(vt.w, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('W (wavelengths)')
        plt.ylabel('Count')
        plt.show()



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_11_0.png



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_11_1.png



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_11_2.png


.. parsed-literal::

    /Users/timcornwell/opt/anaconda3/envs/arlenv37/lib/python3.7/site-packages/ipykernel_launcher.py:23: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_11_4.png


Show the planar nature of the uvw sampling, rotating with hour angle

Create a grid of components and predict each in turn, using the full
phase term including w.

.. code:: ipython3

    npixel = 512
    cellsize=0.001
    facets = 4
    flux = numpy.array([[100.0]])
    vt.data['vis'] *= 0.0
    
    model = create_image_from_visibility(vt, npixel=512, cellsize=0.001, npol=1)
    spacing_pixels = npixel // facets
    log.info('Spacing in pixels = %s' % spacing_pixels)
    spacing = 180.0 * cellsize * spacing_pixels / numpy.pi
    centers = -1.5, -0.5, +0.5, +1.5
    comps=list()
    for iy in centers:
        for ix in centers:
            pra =  int(round(npixel // 2 + ix * spacing_pixels - 1))
            pdec = int(round(npixel // 2 + iy * spacing_pixels - 1))
            sc = pixel_to_skycoord(pra, pdec, model.wcs)
            log.info("Component at (%f, %f) %s" % (pra, pdec, str(sc)))
            comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc, 
                                       polarisation_frame=PolarisationFrame("stokesI"))
            comps.append(comp)
    predict_skycomponent_visibility(vt, comps)


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Spacing in pixels = 128
    Component at (63.000000, 63.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (34.54072794, -54.75874632)>
    Component at (191.000000, 63.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (21.67016023, -55.97155392)>
    Component at (319.000000, 63.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (8.53437599, -55.98101975)>
    Component at (447.000000, 63.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (355.65593443, -54.78677607)>
    Component at (63.000000, 191.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (31.62619218, -47.58256993)>
    Component at (191.000000, 191.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (20.64032824, -48.59686272)>
    Component at (319.000000, 191.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (9.53290242, -48.6047374)>
    Component at (447.000000, 191.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (358.54340538, -47.60612847)>
    Component at (63.000000, 319.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (29.67602513, -40.37841403)>
    Component at (191.000000, 319.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (19.96147534, -41.27364866)>
    Component at (319.000000, 319.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (10.19103897, -41.28057703)>
    Component at (447.000000, 319.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (0.4748226, -40.3992699)>
    Component at (63.000000, 447.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (28.31336506, -33.05766481)>
    Component at (191.000000, 447.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (19.49068833, -33.88364257)>
    Component at (319.000000, 447.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (10.64743131, -33.89002022)>
    Component at (447.000000, 447.000000) <SkyCoord (ICRS): (ra, dec) in deg
        (1.82415538, -33.07694985)>




.. parsed-literal::

    <arl.data_models.memory_data_models.Visibility at 0x105d66710>



Make the dirty image and point spread function using the two-dimensional
approximation:

.. math:: V(u,v,w) =\int I(l,m) e^{2 \pi j (ul+um)} dl dm

Note that the shape of the sources vary with position in the image. This
space-variant property of the PSF arises from the w-term neglected in
the two-dimensional invert.

.. code:: ipython3

    arlexecute.set_client(use_dask=True)


.. parsed-literal::

    Using selector: KqueueSelector
    Using selector: KqueueSelector


.. parsed-literal::

    /Users/timcornwell/opt/anaconda3/envs/arlenv37/lib/python3.7/site-packages/distributed/dashboard/core.py:72: UserWarning: 
    Port 8787 is already in use. 
    Perhaps you already have a cluster running?
    Hosting the diagnostics dashboard on a random port instead.
      warnings.warn("\n" + msg)


.. code:: ipython3

    dirty = create_image_from_visibility(vt, npixel=512, cellsize=0.001, 
                                         polarisation_frame=PolarisationFrame("stokesI"))
    vt = weight_visibility(vt, dirty)
    
    future = invert_list_arlexecute_workflow([vt], [dirty], context='2d')
    dirty, sumwt = arlexecute.compute(future, sync=True)[0]
    
    if doplot:
        show_image(dirty)
    
    print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirty.data.max(), dirty.data.min(), sumwt))
    
    export_image_to_fits(dirty, '%s/imaging-wterm_dirty.fits' % (results_dir))


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]


.. parsed-literal::

    /Users/timcornwell/opt/anaconda3/envs/arlenv37/lib/python3.7/site-packages/distributed/worker.py:3285: UserWarning: Large object of size 2.10 MB detected in task graph: 
      ('getitem-c549e0cacb9af790eb391fd48a1d54ce', <arl. ... -c45882e6197d')
    Consider scattering large objects ahead of time
    with client.scatter to reduce scheduler burden and 
    keep data on workers
    
        future = client.submit(func, big_data)    # bad
    
        big_future = client.scatter(big_data)     # good
        future = client.submit(func, big_future)  # good
      % (format_bytes(len(b)), s)


.. parsed-literal::

    Max, min in dirty image = 49.220748, -8.719588, sumwt = 31701.000000



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_17_3.png


This occurs because the Fourier transform relationship between sky
brightness and visibility is only accurate over small fields of view.

Hence we can make an accurate image by partitioning the image plane into
small regions, treating each separately and then glueing the resulting
partitions into one image. We call this image plane partitioning image
plane faceting.

.. math::

   V(u,v,w) = \sum_{i,j} \frac{1}{\sqrt{1- l_{i,j}^2- m_{i,j}^2}} e^{-2 \pi j (ul_{i,j}+um_{i,j} + w(\sqrt{1-l_{i,j}^2-m_{i,j}^2}-1))}
   \int  I(\Delta l, \Delta m) e^{-2 \pi j (u\Delta l_{i,j}+u \Delta m_{i,j})} dl dm

.. code:: ipython3

    dirtyFacet = create_image_from_visibility(vt, npixel=512, cellsize=0.001, npol=1)
    future = invert_list_arlexecute_workflow([vt], [dirtyFacet], facets=4, context='facets')
    dirtyFacet, sumwt = arlexecute.compute(future, sync=True)[0]
    
    if doplot:
        show_image(dirtyFacet)
    
    print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirtyFacet.data.max(), dirtyFacet.data.min(), sumwt))
    export_image_to_fits(dirtyFacet, '%s/imaging-wterm_dirtyFacet.fits' % (results_dir))


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 102.477593, -11.772359, sumwt = 507216.000000



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_19_1.png


That was the best case. This time, we will not arrange for the
partitions to be centred on the sources.

.. code:: ipython3

    dirtyFacet2 = create_image_from_visibility(vt, npixel=512, cellsize=0.001, npol=1)
    future = invert_list_arlexecute_workflow([vt], [dirtyFacet2], facets=2, context='facets')
    dirtyFacet2, sumwt = arlexecute.compute(future, sync=True)[0]
    
    
    if doplot:
        show_image(dirtyFacet2)
    
    print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirtyFacet2.data.max(), dirtyFacet2.data.min(), sumwt))
    export_image_to_fits(dirtyFacet2, '%s/imaging-wterm_dirtyFacet2.fits' % (results_dir))


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 51.663404, -9.843857, sumwt = 126804.000000



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_21_1.png


Another approach is to partition the visibility data by slices in w. The
measurement equation is approximated as:

.. math:: V(u,v,w) =\sum_i \int   \frac{ I(l,m) e^{-2 \pi j (w_i(\sqrt{1-l^2-m^2}-1))})}{\sqrt{1-l^2-m^2}} e^{-2 \pi j (ul+um)} dl dm

If images constructed from slices in w are added after applying a
w-dependent image plane correction, the w term will be corrected.

The w-dependent w-beam is:

.. code:: ipython3

    if doplot:
        wterm = create_w_term_like(model, phasecentre=vt.phasecentre, w=numpy.max(vt.w))
        show_image(wterm)
        plt.show()


.. parsed-literal::

    create_w_term_image: For w = 169.4, field of view = 0.512000, Fresnel number = 11.10



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_24_1.png


.. code:: ipython3

    dirtywstack = create_image_from_visibility(vt, npixel=512, cellsize=0.001, npol=1)
    future = invert_list_arlexecute_workflow([vt], [dirtywstack], vis_slices=101, context='wstack')
    dirtywstack, sumwt = arlexecute.compute(future, sync=True)[0]
    
    show_image(dirtywstack)
    plt.show()
    
    print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % 
          (dirtywstack.data.max(), dirtywstack.data.min(), sumwt))
    
    export_image_to_fits(dirtywstack, '%s/imaging-wterm_dirty_wstack.fits' % (results_dir))


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_25_1.png


.. parsed-literal::

    Max, min in dirty image = 1682275.134189, -399852.137286, sumwt = 31701.000000


The w-term can also be viewed as a time-variable distortion.
Approximating the array as instantaneously co-planar, we have that w can
be expressed in terms of :math:`u,v`

.. math:: w = a u + b v

Transforming to a new coordinate system:

.. math::  l' = l + a (\sqrt{1-l^2-m^2}-1))

.. math::  m' = m + b (\sqrt{1-l^2-m^2}-1))

Ignoring changes in the normalisation term, we have:

.. math:: V(u,v,w) =\int \frac{I(l',m')}{\sqrt{1-l'^2-m'^2}} e^{-2 \pi j (ul'+um')} dl' dm'

To illustrate this, we will construct images as a function of time. For
comparison, we show difference of each time slice from the best facet
image. Instantaneously the sources are un-distorted but do lie in the
wrong location.

.. code:: ipython3

    for rows in vis_timeslice_iter(vt):
        visslice = create_visibility_from_rows(vt, rows)
        dirtySnapshot = create_image_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1, compress_factor=0.0)
        future = invert_list_arlexecute_workflow([visslice], [dirtySnapshot], context='2d')
        dirtySnapshot, sumwt = arlexecute.compute(future, sync=True)[0]
        
        print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % 
              (dirtySnapshot.data.max(), dirtySnapshot.data.min(), sumwt))
        if doplot:
            dirtySnapshot.data -= dirtyFacet.data
            show_image(dirtySnapshot)
            plt.title("Hour angle %.2f hours" % (numpy.average(visslice.time) * 12.0 / 43200.0))
            plt.show()


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 193.168337 wavelengths
    create_image_from_visibility: Critical cellsize = 0.002588 radians, 0.148305 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 95.641822, -16.727650, sumwt = 4216.610661



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_27_1.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 231.753249 wavelengths
    create_image_from_visibility: Critical cellsize = 0.002157 radians, 0.123614 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 100.519607, -17.807714, sumwt = 4508.422216



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_27_3.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 255.914039 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001954 radians, 0.111943 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 99.690697, -17.608646, sumwt = 4735.435580



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_27_5.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 102.909868, -19.624505, sumwt = 4806.769444



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_27_7.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 253.664251 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001971 radians, 0.112936 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 101.931847, -18.677038, sumwt = 4732.772801



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_27_9.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 236.819632 wavelengths
    create_image_from_visibility: Critical cellsize = 0.002111 radians, 0.120969 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 102.692951, -18.823002, sumwt = 4524.935947



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_27_11.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 204.762355 wavelengths
    create_image_from_visibility: Critical cellsize = 0.002442 radians, 0.139908 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    Max, min in dirty image = 98.916621, -17.862308, sumwt = 4176.053350



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_27_13.png


This timeslice imaging leads to a straightforward algorithm in which we
correct each time slice and then sum the resulting timeslices.

.. code:: ipython3

    dirtyTimeslice = create_image_from_visibility(vt, npixel=512, cellsize=0.001, npol=1)
    future = invert_list_arlexecute_workflow([vt], [dirtyTimeslice], vis_slices=vis_timeslices(vt, 'auto'),
                                           padding=2, context='timeslice')
    dirtyTimeslice, sumwt = arlexecute.compute(future, sync=True)[0]
    
    
    show_image(dirtyTimeslice)
    plt.show()
    
    print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % 
          (dirtyTimeslice.data.max(), dirtyTimeslice.data.min(), sumwt))
    
    export_image_to_fits(dirtyTimeslice, '%s/imaging-wterm_dirty_Timeslice.fits' % (results_dir))


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_29_1.png


.. parsed-literal::

    Max, min in dirty image = 3557752.468175, -625759.676690, sumwt = 31701.000000


Finally we try w-projection. For a fixed w, the measurement equation can
be stated as as a convolution in Fourier space.

.. math:: V(u,v,w) =G_w(u,v) \ast \int \frac{I(l,m)}{\sqrt{1-l^2-m^2}} e^{-2 \pi j (ul+um)} dl dm

where the convolution function is:

.. math:: G_w(u,v) = \int \frac{1}{\sqrt{1-l^2-m^2}} e^{-2 \pi j (ul+um + w(\sqrt{1-l^2-m^2}-1))} dl dm

Hence when gridding, we can use the transform of the w beam to correct
this effect while gridding.

.. code:: ipython3

    dirtyWProjection = create_image_from_visibility(vt, npixel=512, cellsize=0.001, npol=1)
    
    gcfcf = create_awterm_convolutionfunction(model, nw=101, wstep=800.0/101, oversampling=8, 
                                                        support=60,
                                                        use_aaf=True)
        
    future = invert_list_arlexecute_workflow([vt], [dirtyWProjection], context='2d', gcfcf=[gcfcf])
    
    dirtyWProjection, sumwt = arlexecute.compute(future, sync=True)[0]
    
    if doplot:
        show_image(dirtyWProjection)
    
    print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirtyWProjection.data.max(), 
                                                                 dirtyWProjection.data.min(), sumwt))
    export_image_to_fits(dirtyWProjection, '%s/imaging-wterm_dirty_WProjection.fits' % (results_dir))



.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 9999999.9999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 512, 512]
    create_w_term_image: For w = -396.0, field of view = 0.512000, Fresnel number = 25.95
    create_w_term_image: For w = -388.1, field of view = 0.512000, Fresnel number = 25.44
    create_w_term_image: For w = -380.2, field of view = 0.512000, Fresnel number = 24.92
    create_w_term_image: For w = -372.3, field of view = 0.512000, Fresnel number = 24.40
    create_w_term_image: For w = -364.4, field of view = 0.512000, Fresnel number = 23.88
    create_w_term_image: For w = -356.4, field of view = 0.512000, Fresnel number = 23.36
    create_w_term_image: For w = -348.5, field of view = 0.512000, Fresnel number = 22.84
    create_w_term_image: For w = -340.6, field of view = 0.512000, Fresnel number = 22.32
    create_w_term_image: For w = -332.7, field of view = 0.512000, Fresnel number = 21.80
    create_w_term_image: For w = -324.8, field of view = 0.512000, Fresnel number = 21.28
    create_w_term_image: For w = -316.8, field of view = 0.512000, Fresnel number = 20.76
    create_w_term_image: For w = -308.9, field of view = 0.512000, Fresnel number = 20.24
    create_w_term_image: For w = -301.0, field of view = 0.512000, Fresnel number = 19.73
    create_w_term_image: For w = -293.1, field of view = 0.512000, Fresnel number = 19.21
    create_w_term_image: For w = -285.1, field of view = 0.512000, Fresnel number = 18.69
    create_w_term_image: For w = -277.2, field of view = 0.512000, Fresnel number = 18.17
    create_w_term_image: For w = -269.3, field of view = 0.512000, Fresnel number = 17.65
    create_w_term_image: For w = -261.4, field of view = 0.512000, Fresnel number = 17.13
    create_w_term_image: For w = -253.5, field of view = 0.512000, Fresnel number = 16.61
    create_w_term_image: For w = -245.5, field of view = 0.512000, Fresnel number = 16.09
    create_w_term_image: For w = -237.6, field of view = 0.512000, Fresnel number = 15.57
    create_w_term_image: For w = -229.7, field of view = 0.512000, Fresnel number = 15.05
    create_w_term_image: For w = -221.8, field of view = 0.512000, Fresnel number = 14.53
    create_w_term_image: For w = -213.9, field of view = 0.512000, Fresnel number = 14.02
    create_w_term_image: For w = -205.9, field of view = 0.512000, Fresnel number = 13.50
    create_w_term_image: For w = -198.0, field of view = 0.512000, Fresnel number = 12.98
    create_w_term_image: For w = -190.1, field of view = 0.512000, Fresnel number = 12.46
    create_w_term_image: For w = -182.2, field of view = 0.512000, Fresnel number = 11.94
    create_w_term_image: For w = -174.3, field of view = 0.512000, Fresnel number = 11.42
    create_w_term_image: For w = -166.3, field of view = 0.512000, Fresnel number = 10.90
    create_w_term_image: For w = -158.4, field of view = 0.512000, Fresnel number = 10.38
    create_w_term_image: For w = -150.5, field of view = 0.512000, Fresnel number = 9.86
    create_w_term_image: For w = -142.6, field of view = 0.512000, Fresnel number = 9.34
    create_w_term_image: For w = -134.7, field of view = 0.512000, Fresnel number = 8.82
    create_w_term_image: For w = -126.7, field of view = 0.512000, Fresnel number = 8.31
    create_w_term_image: For w = -118.8, field of view = 0.512000, Fresnel number = 7.79
    create_w_term_image: For w = -110.9, field of view = 0.512000, Fresnel number = 7.27
    create_w_term_image: For w = -103.0, field of view = 0.512000, Fresnel number = 6.75
    create_w_term_image: For w = -95.0, field of view = 0.512000, Fresnel number = 6.23
    create_w_term_image: For w = -87.1, field of view = 0.512000, Fresnel number = 5.71
    create_w_term_image: For w = -79.2, field of view = 0.512000, Fresnel number = 5.19
    create_w_term_image: For w = -71.3, field of view = 0.512000, Fresnel number = 4.67
    create_w_term_image: For w = -63.4, field of view = 0.512000, Fresnel number = 4.15
    create_w_term_image: For w = -55.4, field of view = 0.512000, Fresnel number = 3.63
    create_w_term_image: For w = -47.5, field of view = 0.512000, Fresnel number = 3.11
    create_w_term_image: For w = -39.6, field of view = 0.512000, Fresnel number = 2.60
    create_w_term_image: For w = -31.7, field of view = 0.512000, Fresnel number = 2.08
    create_w_term_image: For w = -23.8, field of view = 0.512000, Fresnel number = 1.56
    create_w_term_image: For w = -15.8, field of view = 0.512000, Fresnel number = 1.04
    create_w_term_image: For w = -7.9, field of view = 0.512000, Fresnel number = 0.52
    create_w_term_image: For w = 0.0, field of view = 0.512000, Fresnel number = 0.00
    create_w_term_image: For w = 7.9, field of view = 0.512000, Fresnel number = 0.52
    create_w_term_image: For w = 15.8, field of view = 0.512000, Fresnel number = 1.04
    create_w_term_image: For w = 23.8, field of view = 0.512000, Fresnel number = 1.56
    create_w_term_image: For w = 31.7, field of view = 0.512000, Fresnel number = 2.08
    create_w_term_image: For w = 39.6, field of view = 0.512000, Fresnel number = 2.60
    create_w_term_image: For w = 47.5, field of view = 0.512000, Fresnel number = 3.11
    create_w_term_image: For w = 55.4, field of view = 0.512000, Fresnel number = 3.63
    create_w_term_image: For w = 63.4, field of view = 0.512000, Fresnel number = 4.15
    create_w_term_image: For w = 71.3, field of view = 0.512000, Fresnel number = 4.67
    create_w_term_image: For w = 79.2, field of view = 0.512000, Fresnel number = 5.19
    create_w_term_image: For w = 87.1, field of view = 0.512000, Fresnel number = 5.71
    create_w_term_image: For w = 95.0, field of view = 0.512000, Fresnel number = 6.23
    create_w_term_image: For w = 103.0, field of view = 0.512000, Fresnel number = 6.75
    create_w_term_image: For w = 110.9, field of view = 0.512000, Fresnel number = 7.27
    create_w_term_image: For w = 118.8, field of view = 0.512000, Fresnel number = 7.79
    create_w_term_image: For w = 126.7, field of view = 0.512000, Fresnel number = 8.31
    create_w_term_image: For w = 134.7, field of view = 0.512000, Fresnel number = 8.82
    create_w_term_image: For w = 142.6, field of view = 0.512000, Fresnel number = 9.34
    create_w_term_image: For w = 150.5, field of view = 0.512000, Fresnel number = 9.86
    create_w_term_image: For w = 158.4, field of view = 0.512000, Fresnel number = 10.38
    create_w_term_image: For w = 166.3, field of view = 0.512000, Fresnel number = 10.90
    create_w_term_image: For w = 174.3, field of view = 0.512000, Fresnel number = 11.42
    create_w_term_image: For w = 182.2, field of view = 0.512000, Fresnel number = 11.94
    create_w_term_image: For w = 190.1, field of view = 0.512000, Fresnel number = 12.46
    create_w_term_image: For w = 198.0, field of view = 0.512000, Fresnel number = 12.98
    create_w_term_image: For w = 205.9, field of view = 0.512000, Fresnel number = 13.50
    create_w_term_image: For w = 213.9, field of view = 0.512000, Fresnel number = 14.02
    create_w_term_image: For w = 221.8, field of view = 0.512000, Fresnel number = 14.53
    create_w_term_image: For w = 229.7, field of view = 0.512000, Fresnel number = 15.05
    create_w_term_image: For w = 237.6, field of view = 0.512000, Fresnel number = 15.57
    create_w_term_image: For w = 245.5, field of view = 0.512000, Fresnel number = 16.09
    create_w_term_image: For w = 253.5, field of view = 0.512000, Fresnel number = 16.61
    create_w_term_image: For w = 261.4, field of view = 0.512000, Fresnel number = 17.13
    create_w_term_image: For w = 269.3, field of view = 0.512000, Fresnel number = 17.65
    create_w_term_image: For w = 277.2, field of view = 0.512000, Fresnel number = 18.17
    create_w_term_image: For w = 285.1, field of view = 0.512000, Fresnel number = 18.69
    create_w_term_image: For w = 293.1, field of view = 0.512000, Fresnel number = 19.21
    create_w_term_image: For w = 301.0, field of view = 0.512000, Fresnel number = 19.73
    create_w_term_image: For w = 308.9, field of view = 0.512000, Fresnel number = 20.24
    create_w_term_image: For w = 316.8, field of view = 0.512000, Fresnel number = 20.76
    create_w_term_image: For w = 324.8, field of view = 0.512000, Fresnel number = 21.28
    create_w_term_image: For w = 332.7, field of view = 0.512000, Fresnel number = 21.80
    create_w_term_image: For w = 340.6, field of view = 0.512000, Fresnel number = 22.32
    create_w_term_image: For w = 348.5, field of view = 0.512000, Fresnel number = 22.84
    create_w_term_image: For w = 356.4, field of view = 0.512000, Fresnel number = 23.36
    create_w_term_image: For w = 364.4, field of view = 0.512000, Fresnel number = 23.88
    create_w_term_image: For w = 372.3, field of view = 0.512000, Fresnel number = 24.40
    create_w_term_image: For w = 380.2, field of view = 0.512000, Fresnel number = 24.92
    create_w_term_image: For w = 388.1, field of view = 0.512000, Fresnel number = 25.44
    create_w_term_image: For w = 396.0, field of view = 0.512000, Fresnel number = 25.95
    Max, min in dirty image = 100.225951, -11.016452, sumwt = 31701.000000



.. image:: imaging-wterm_arlexecute_files/imaging-wterm_arlexecute_31_1.png


