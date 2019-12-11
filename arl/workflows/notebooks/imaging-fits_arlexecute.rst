Demonstrate full circle wide field imaging
------------------------------------------

This include prediction of components, inversion, point source fitting.
We will compare the output images with the input models, looking for
closeness in flux and position.

.. code:: ipython3

    %matplotlib inline
    
    import os
    import sys
    
    sys.path.append(os.path.join('..', '..'))
    
    from data_models.parameters import arl_path
    results_dir = arl_path('test_results')
    
    from matplotlib import pylab
    
    pylab.rcParams['figure.figsize'] = (10.0, 10.0)
    pylab.rcParams['image.cmap'] = 'rainbow'
    
    from matplotlib import pyplot as plt
    
    import numpy
    
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.wcs.utils import pixel_to_skycoord
    
    from data_models import PolarisationFrame
    
    from processing_components import create_visibility, sum_visibility, vis_timeslices,create_skycomponent, \
        find_skycomponents, find_nearest_skycomponent, insert_skycomponent, show_image, export_image_to_fits, \
        qa_image, smooth_image, create_named_configuration, advise_wide_field, create_image_from_visibility, \
        predict_skycomponent_visibility, create_awterm_convolutionfunction, apply_bounding_box_convolutionfunction
    
    # Use workflows for imaging
    from wrappers.arlexecute.execution_support.arlexecute import arlexecute
    
    from workflows import  imaging_contexts, predict_list_arlexecute_workflow, \
        invert_list_arlexecute_workflow
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    
    mpl_logger = logging.getLogger("matplotlib") 
    mpl_logger.setLevel(logging.WARNING) 


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


Use Dask

.. code:: ipython3

    arlexecute.set_client(use_dask=True)


.. parsed-literal::

    Using selector: KqueueSelector
    Using selector: KqueueSelector


We create the visibility. This just makes the uvw, time, antenna1,
antenna2, weight columns in a table

.. code:: ipython3

    times = numpy.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) * (numpy.pi / 12.0)
    frequency = numpy.array([1e8])
    channel_bandwidth = numpy.array([1e6])
    reffrequency = numpy.max(frequency)
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                           weight=1.0, phasecentre=phasecentre, 
                           polarisation_frame=PolarisationFrame('stokesI'))


.. parsed-literal::

    create_visibility: 95865 rows, 0.010 GB
    create_visibility: flagged 0/95865 visibilities below elevation limit 0.261799 (rad)


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


Fill in the visibility with exact calculation of a number of point
sources

.. code:: ipython3

    vt.data['vis'] *= 0.0
    npixel=256
    
    model = create_image_from_visibility(vt, npixel=npixel, cellsize=0.001, nchan=1, 
                                         polarisation_frame=PolarisationFrame('stokesI'))
    centre = model.wcs.wcs.crpix-1
    spacing_pixels = npixel // 8
    log.info('Spacing in pixels = %s' % spacing_pixels)
    spacing = model.wcs.wcs.cdelt * spacing_pixels
    locations = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    
    original_comps = []
    # We calculate the source positions in pixels and then calculate the
    # world coordinates to put in the skycomponent description
    for iy in locations:
        for ix in locations:
            if ix >= iy:
                p = int(round(centre[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                    int(round(centre[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                sc = pixel_to_skycoord(p[0], p[1], model.wcs)
                log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))
                flux = numpy.array([[100.0 + 2.0 * ix + iy * 20.0]])
                comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc, 
                                           polarisation_frame=PolarisationFrame('stokesI'))
                original_comps.append(comp)
                insert_skycomponent(model, comp)
    
    predict_skycomponent_visibility(vt, original_comps)
    
            
    cmodel = smooth_image(model) 
    show_image(cmodel)
    plt.title("Smoothed model image")
    plt.show()


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 999999.99999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 256, 256]
    Spacing in pixels = 32
    Component at (240.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (4.74378292, -51.0209895)>
    insert_skycomponent: Using insert method Nearest
    Component at (208.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (7.66144755, -51.22149637)>
    insert_skycomponent: Using insert method Nearest
    Component at (176.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (10.5917361, -51.35530369)>
    insert_skycomponent: Using insert method Nearest
    Component at (144.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (13.52971731, -51.42224945)>
    insert_skycomponent: Using insert method Nearest
    Component at (112.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (16.47028269, -51.42224945)>
    insert_skycomponent: Using insert method Nearest
    Component at (80.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (19.4082639, -51.35530369)>
    insert_skycomponent: Using insert method Nearest
    Component at (48.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (22.33855245, -51.22149637)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 16.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (25.25621708, -51.0209895)>
    insert_skycomponent: Using insert method Nearest
    Component at (208.000000, 48.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (7.94046978, -49.38802398)>
    insert_skycomponent: Using insert method Nearest
    Component at (176.000000, 48.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (10.76004686, -49.51635065)>
    insert_skycomponent: Using insert method Nearest
    Component at (144.000000, 48.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (13.58597172, -49.5805407)>
    insert_skycomponent: Using insert method Nearest
    Component at (112.000000, 48.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (16.41402828, -49.5805407)>
    insert_skycomponent: Using insert method Nearest
    Component at (80.000000, 48.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (19.23995314, -49.51635065)>
    insert_skycomponent: Using insert method Nearest
    Component at (48.000000, 48.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (22.05953022, -49.38802398)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 48.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (24.86868384, -49.19566221)>
    insert_skycomponent: Using insert method Nearest
    Component at (176.000000, 80.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (10.91156133, -47.68176393)>
    insert_skycomponent: Using insert method Nearest
    Component at (144.000000, 80.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (13.63659973, -47.74353279)>
    insert_skycomponent: Using insert method Nearest
    Component at (112.000000, 80.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (16.36340027, -47.74353279)>
    insert_skycomponent: Using insert method Nearest
    Component at (80.000000, 80.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (19.08843867, -47.68176393)>
    insert_skycomponent: Using insert method Nearest
    Component at (48.000000, 80.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (21.80822877, -47.55825467)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 80.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (24.51939763, -47.3730574)>
    insert_skycomponent: Using insert method Nearest
    Component at (144.000000, 112.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (13.68235336, -45.90931658)>
    insert_skycomponent: Using insert method Nearest
    Component at (112.000000, 112.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (16.31764664, -45.90931658)>
    insert_skycomponent: Using insert method Nearest
    Component at (80.000000, 112.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (18.95148298, -45.84967813)>
    insert_skycomponent: Using insert method Nearest
    Component at (48.000000, 112.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (21.58097973, -45.73040913)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 112.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (24.20334715, -45.55152143)>
    insert_skycomponent: Using insert method Nearest
    Component at (112.000000, 144.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (16.27614945, -44.07600819)>
    insert_skycomponent: Using insert method Nearest
    Component at (80.000000, 144.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (18.82724608, -44.01824481)>
    insert_skycomponent: Using insert method Nearest
    Component at (48.000000, 144.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (21.37476195, -43.90270873)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 144.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (23.91639614, -43.72937783)>
    insert_skycomponent: Using insert method Nearest
    Component at (80.000000, 176.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (18.71420256, -42.18561677)>
    insert_skycomponent: Using insert method Nearest
    Component at (48.000000, 176.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (21.18706763, -42.07336315)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 176.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (23.65510465, -41.90491964)>
    insert_skycomponent: Using insert method Nearest
    Component at (48.000000, 208.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (21.01580068, -40.24055692)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 208.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (23.41659285, -40.07639948)>
    insert_skycomponent: Using insert method Nearest
    Component at (16.000000, 240.000000) [0-rel] <SkyCoord (ICRS): (ra, dec) in deg
        (23.19843619, -38.24201755)>
    insert_skycomponent: Using insert method Nearest



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_11_1.png


Check that the skycoordinate and image coordinate system are consistent
by finding the point sources.

.. code:: ipython3

    comps = find_skycomponents(cmodel, fwhm=1.0, threshold=10.0, npixels=5)
    plt.clf()
    for i in range(len(comps)):
        ocomp, sep = find_nearest_skycomponent(comps[i].direction, original_comps)
        plt.plot((comps[i].direction.ra.value  - ocomp.direction.ra.value)/cmodel.wcs.wcs.cdelt[0], 
                 (comps[i].direction.dec.value - ocomp.direction.dec.value)/cmodel.wcs.wcs.cdelt[1], 
                 '.', color='r')  
    
    plt.xlabel('delta RA (pixels)')
    plt.ylabel('delta DEC (pixels)')
    plt.title("Recovered - Original position offsets")
    plt.show()


.. parsed-literal::

    find_skycomponents: Finding components in Image by segmentation
    find_skycomponents: Identified 36 segments



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_13_1.png


Make the convolution function

.. code:: ipython3

    wstep = 8.0
    nw = int(1.1 * 800/wstep)
        
    gcfcf = create_awterm_convolutionfunction(model, nw=110, wstep=8, oversampling=8, 
                                                        support=60,
                                                        use_aaf=True)
        
    cf=gcfcf[1]
    print(cf.data.shape)
    plt.clf()
    plt.imshow(numpy.real(cf.data[0,0,0,0,0,:,:]))
    plt.title(str(numpy.max(numpy.abs(cf.data[0,0,0,0,0,:,:]))))
    plt.show()
        
    cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=1e-3)
    print(cf_clipped.data.shape)
    gcfcf_clipped=(gcfcf[0], cf_clipped)
        
    plt.clf()
    plt.imshow(numpy.real(cf_clipped.data[0,0,0,0,0,:,:]))
    plt.title(str(numpy.max(numpy.abs(cf_clipped.data[0,0,0,0,0,:,:]))))
    plt.show()



.. parsed-literal::

    create_w_term_image: For w = -440.0, field of view = 0.256000, Fresnel number = 7.21
    create_w_term_image: For w = -432.0, field of view = 0.256000, Fresnel number = 7.08
    create_w_term_image: For w = -424.0, field of view = 0.256000, Fresnel number = 6.95
    create_w_term_image: For w = -416.0, field of view = 0.256000, Fresnel number = 6.82
    create_w_term_image: For w = -408.0, field of view = 0.256000, Fresnel number = 6.68
    create_w_term_image: For w = -400.0, field of view = 0.256000, Fresnel number = 6.55
    create_w_term_image: For w = -392.0, field of view = 0.256000, Fresnel number = 6.42
    create_w_term_image: For w = -384.0, field of view = 0.256000, Fresnel number = 6.29
    create_w_term_image: For w = -376.0, field of view = 0.256000, Fresnel number = 6.16
    create_w_term_image: For w = -368.0, field of view = 0.256000, Fresnel number = 6.03
    create_w_term_image: For w = -360.0, field of view = 0.256000, Fresnel number = 5.90
    create_w_term_image: For w = -352.0, field of view = 0.256000, Fresnel number = 5.77
    create_w_term_image: For w = -344.0, field of view = 0.256000, Fresnel number = 5.64
    create_w_term_image: For w = -336.0, field of view = 0.256000, Fresnel number = 5.51
    create_w_term_image: For w = -328.0, field of view = 0.256000, Fresnel number = 5.37
    create_w_term_image: For w = -320.0, field of view = 0.256000, Fresnel number = 5.24
    create_w_term_image: For w = -312.0, field of view = 0.256000, Fresnel number = 5.11
    create_w_term_image: For w = -304.0, field of view = 0.256000, Fresnel number = 4.98
    create_w_term_image: For w = -296.0, field of view = 0.256000, Fresnel number = 4.85
    create_w_term_image: For w = -288.0, field of view = 0.256000, Fresnel number = 4.72
    create_w_term_image: For w = -280.0, field of view = 0.256000, Fresnel number = 4.59
    create_w_term_image: For w = -272.0, field of view = 0.256000, Fresnel number = 4.46
    create_w_term_image: For w = -264.0, field of view = 0.256000, Fresnel number = 4.33
    create_w_term_image: For w = -256.0, field of view = 0.256000, Fresnel number = 4.19
    create_w_term_image: For w = -248.0, field of view = 0.256000, Fresnel number = 4.06
    create_w_term_image: For w = -240.0, field of view = 0.256000, Fresnel number = 3.93
    create_w_term_image: For w = -232.0, field of view = 0.256000, Fresnel number = 3.80
    create_w_term_image: For w = -224.0, field of view = 0.256000, Fresnel number = 3.67
    create_w_term_image: For w = -216.0, field of view = 0.256000, Fresnel number = 3.54
    create_w_term_image: For w = -208.0, field of view = 0.256000, Fresnel number = 3.41
    create_w_term_image: For w = -200.0, field of view = 0.256000, Fresnel number = 3.28
    create_w_term_image: For w = -192.0, field of view = 0.256000, Fresnel number = 3.15
    create_w_term_image: For w = -184.0, field of view = 0.256000, Fresnel number = 3.01
    create_w_term_image: For w = -176.0, field of view = 0.256000, Fresnel number = 2.88
    create_w_term_image: For w = -168.0, field of view = 0.256000, Fresnel number = 2.75
    create_w_term_image: For w = -160.0, field of view = 0.256000, Fresnel number = 2.62
    create_w_term_image: For w = -152.0, field of view = 0.256000, Fresnel number = 2.49
    create_w_term_image: For w = -144.0, field of view = 0.256000, Fresnel number = 2.36
    create_w_term_image: For w = -136.0, field of view = 0.256000, Fresnel number = 2.23
    create_w_term_image: For w = -128.0, field of view = 0.256000, Fresnel number = 2.10
    create_w_term_image: For w = -120.0, field of view = 0.256000, Fresnel number = 1.97
    create_w_term_image: For w = -112.0, field of view = 0.256000, Fresnel number = 1.84
    create_w_term_image: For w = -104.0, field of view = 0.256000, Fresnel number = 1.70
    create_w_term_image: For w = -96.0, field of view = 0.256000, Fresnel number = 1.57
    create_w_term_image: For w = -88.0, field of view = 0.256000, Fresnel number = 1.44
    create_w_term_image: For w = -80.0, field of view = 0.256000, Fresnel number = 1.31
    create_w_term_image: For w = -72.0, field of view = 0.256000, Fresnel number = 1.18
    create_w_term_image: For w = -64.0, field of view = 0.256000, Fresnel number = 1.05
    create_w_term_image: For w = -56.0, field of view = 0.256000, Fresnel number = 0.92
    create_w_term_image: For w = -48.0, field of view = 0.256000, Fresnel number = 0.79
    create_w_term_image: For w = -40.0, field of view = 0.256000, Fresnel number = 0.66
    create_w_term_image: For w = -32.0, field of view = 0.256000, Fresnel number = 0.52
    create_w_term_image: For w = -24.0, field of view = 0.256000, Fresnel number = 0.39
    create_w_term_image: For w = -16.0, field of view = 0.256000, Fresnel number = 0.26
    create_w_term_image: For w = -8.0, field of view = 0.256000, Fresnel number = 0.13
    create_w_term_image: For w = 0.0, field of view = 0.256000, Fresnel number = 0.00
    create_w_term_image: For w = 8.0, field of view = 0.256000, Fresnel number = 0.13
    create_w_term_image: For w = 16.0, field of view = 0.256000, Fresnel number = 0.26
    create_w_term_image: For w = 24.0, field of view = 0.256000, Fresnel number = 0.39
    create_w_term_image: For w = 32.0, field of view = 0.256000, Fresnel number = 0.52
    create_w_term_image: For w = 40.0, field of view = 0.256000, Fresnel number = 0.66
    create_w_term_image: For w = 48.0, field of view = 0.256000, Fresnel number = 0.79
    create_w_term_image: For w = 56.0, field of view = 0.256000, Fresnel number = 0.92
    create_w_term_image: For w = 64.0, field of view = 0.256000, Fresnel number = 1.05
    create_w_term_image: For w = 72.0, field of view = 0.256000, Fresnel number = 1.18
    create_w_term_image: For w = 80.0, field of view = 0.256000, Fresnel number = 1.31
    create_w_term_image: For w = 88.0, field of view = 0.256000, Fresnel number = 1.44
    create_w_term_image: For w = 96.0, field of view = 0.256000, Fresnel number = 1.57
    create_w_term_image: For w = 104.0, field of view = 0.256000, Fresnel number = 1.70
    create_w_term_image: For w = 112.0, field of view = 0.256000, Fresnel number = 1.84
    create_w_term_image: For w = 120.0, field of view = 0.256000, Fresnel number = 1.97
    create_w_term_image: For w = 128.0, field of view = 0.256000, Fresnel number = 2.10
    create_w_term_image: For w = 136.0, field of view = 0.256000, Fresnel number = 2.23
    create_w_term_image: For w = 144.0, field of view = 0.256000, Fresnel number = 2.36
    create_w_term_image: For w = 152.0, field of view = 0.256000, Fresnel number = 2.49
    create_w_term_image: For w = 160.0, field of view = 0.256000, Fresnel number = 2.62
    create_w_term_image: For w = 168.0, field of view = 0.256000, Fresnel number = 2.75
    create_w_term_image: For w = 176.0, field of view = 0.256000, Fresnel number = 2.88
    create_w_term_image: For w = 184.0, field of view = 0.256000, Fresnel number = 3.01
    create_w_term_image: For w = 192.0, field of view = 0.256000, Fresnel number = 3.15
    create_w_term_image: For w = 200.0, field of view = 0.256000, Fresnel number = 3.28
    create_w_term_image: For w = 208.0, field of view = 0.256000, Fresnel number = 3.41
    create_w_term_image: For w = 216.0, field of view = 0.256000, Fresnel number = 3.54
    create_w_term_image: For w = 224.0, field of view = 0.256000, Fresnel number = 3.67
    create_w_term_image: For w = 232.0, field of view = 0.256000, Fresnel number = 3.80
    create_w_term_image: For w = 240.0, field of view = 0.256000, Fresnel number = 3.93
    create_w_term_image: For w = 248.0, field of view = 0.256000, Fresnel number = 4.06
    create_w_term_image: For w = 256.0, field of view = 0.256000, Fresnel number = 4.19
    create_w_term_image: For w = 264.0, field of view = 0.256000, Fresnel number = 4.33
    create_w_term_image: For w = 272.0, field of view = 0.256000, Fresnel number = 4.46
    create_w_term_image: For w = 280.0, field of view = 0.256000, Fresnel number = 4.59
    create_w_term_image: For w = 288.0, field of view = 0.256000, Fresnel number = 4.72
    create_w_term_image: For w = 296.0, field of view = 0.256000, Fresnel number = 4.85
    create_w_term_image: For w = 304.0, field of view = 0.256000, Fresnel number = 4.98
    create_w_term_image: For w = 312.0, field of view = 0.256000, Fresnel number = 5.11
    create_w_term_image: For w = 320.0, field of view = 0.256000, Fresnel number = 5.24
    create_w_term_image: For w = 328.0, field of view = 0.256000, Fresnel number = 5.37
    create_w_term_image: For w = 336.0, field of view = 0.256000, Fresnel number = 5.51
    create_w_term_image: For w = 344.0, field of view = 0.256000, Fresnel number = 5.64
    create_w_term_image: For w = 352.0, field of view = 0.256000, Fresnel number = 5.77
    create_w_term_image: For w = 360.0, field of view = 0.256000, Fresnel number = 5.90
    create_w_term_image: For w = 368.0, field of view = 0.256000, Fresnel number = 6.03
    create_w_term_image: For w = 376.0, field of view = 0.256000, Fresnel number = 6.16
    create_w_term_image: For w = 384.0, field of view = 0.256000, Fresnel number = 6.29
    create_w_term_image: For w = 392.0, field of view = 0.256000, Fresnel number = 6.42
    create_w_term_image: For w = 400.0, field of view = 0.256000, Fresnel number = 6.55
    create_w_term_image: For w = 408.0, field of view = 0.256000, Fresnel number = 6.68
    create_w_term_image: For w = 416.0, field of view = 0.256000, Fresnel number = 6.82
    create_w_term_image: For w = 424.0, field of view = 0.256000, Fresnel number = 6.95
    create_w_term_image: For w = 432.0, field of view = 0.256000, Fresnel number = 7.08
    (1, 1, 110, 8, 8, 60, 60)



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_15_1.png


.. parsed-literal::

    (1, 1, 110, 8, 8, 34, 34)



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_15_3.png


Predict the visibility using the different approaches.

.. code:: ipython3

    contexts = imaging_contexts().keys()
    print(contexts)


.. parsed-literal::

    dict_keys(['2d', 'ng', 'wprojection', 'wsnapshots', 'facets', 'facets_timeslice', 'facets_wstack', 'timeslice', 'wstack'])


.. code:: ipython3

    print(gcfcf_clipped[1])


.. parsed-literal::

    Convolution function:
    	Shape: (1, 1, 110, 8, 8, 34, 34)
    	Grid WCS: WCS Transformation
    
    This transformation has 7 pixel and 7 world dimensions
    
    Array shape (Numpy order): None
    
    Pixel Dim  Data size  Bounds
            0       None  None
            1       None  None
            2       None  None
            3       None  None
            4       None  None
            5       None  None
            6       None  None
    
    World Dim  Physical Type  Units
            0  None           unknown
            1  None           unknown
            2  None           unknown
            3  None           unknown
            4  None           unknown
            5  None           unknown
            6  em.freq        Hz
    
    Correlation between pixel and world axes:
    
                           Pixel Dim
    World Dim    0    1    2    3    4    5    6
            0  yes   no   no   no   no   no   no
            1   no  yes   no   no   no   no   no
            2   no   no  yes   no   no   no   no
            3   no   no   no  yes   no   no   no
            4   no   no   no   no  yes   no   no
            5   no   no   no   no   no  yes   no
            6   no   no   no   no   no   no  yes
    	Projection WCS: WCS Transformation
    
    This transformation has 4 pixel and 4 world dimensions
    
    Array shape (Numpy order): None
    
    Pixel Dim  Data size  Bounds
            0       None  None
            1       None  None
            2       None  None
            3       None  None
    
    World Dim  Physical Type  Units
            0  pos.eq.ra      deg
            1  pos.eq.dec     deg
            2  None           unknown
            3  em.freq        Hz
    
    Correlation between pixel and world axes:
    
                   Pixel Dim
    World Dim    0    1    2    3
            0  yes  yes   no   no
            1  yes  yes   no   no
            2   no   no  yes   no
            3   no   no   no  yes
    	Polarisation frame: stokesI
    


.. code:: ipython3

    contexts = ['2d', 'facets', 'timeslice', 'wstack', 'wprojection']
    
    for context in contexts:
        
        print('Processing context %s' % context)
       
        vtpredict_list =[create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
            weight=1.0, phasecentre=phasecentre, polarisation_frame=PolarisationFrame('stokesI'))]
        model_list = [model]
        vtpredict_list = arlexecute.compute(vtpredict_list, sync=True)
        vtpredict_list = arlexecute.scatter(vtpredict_list)
     
        if context == 'wprojection':       
            future = predict_list_arlexecute_workflow(vtpredict_list, model_list, context='2d', gcfcf=[gcfcf_clipped])
        
        elif context == 'facets':
            future = predict_list_arlexecute_workflow(vtpredict_list, model_list, context=context, facets=8)
            
        elif context == 'timeslice':
            future = predict_list_arlexecute_workflow(vtpredict_list, model_list, context=context, vis_slices=vis_timeslices(
                vtpredict, 'auto'))
    
        elif context == 'wstack':
            future = predict_list_arlexecute_workflow(vtpredict_list, model_list, context=context, vis_slices=31)
    
        else:
            future = predict_list_arlexecute_workflow(vtpredict_list, model_list, context=context)
            
        vtpredict_list = arlexecute.compute(future, sync=True)
            
        vtpredict = vtpredict_list[0]
    
        uvdist = numpy.sqrt(vt.data['uvw'][:, 0] ** 2 + vt.data['uvw'][:, 1] ** 2)
        plt.clf()
        plt.plot(uvdist, numpy.abs(vt.data['vis'][:]), '.', color='r', label="DFT")
        
        plt.plot(uvdist, numpy.abs(vtpredict.data['vis'][:]), '.', color='b', label=context)
        plt.plot(uvdist, numpy.abs(vtpredict.data['vis'][:] - vt.data['vis'][:]), '.', color='g', label="Residual")
        plt.xlabel('uvdist')
        plt.ylabel('Amp Visibility')
        plt.legend()
        plt.show()



.. parsed-literal::

    Processing context 2d
    create_visibility: 95865 rows, 0.010 GB
    create_visibility: flagged 0/95865 visibilities below elevation limit 0.261799 (rad)


.. parsed-literal::

    /Users/timcornwell/opt/anaconda3/envs/arlenv37/lib/python3.7/site-packages/IPython/core/pylabtools.py:128: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      fig.canvas.print_figure(bytes_io, **kw)



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_19_2.png


.. parsed-literal::

    Processing context facets
    create_visibility: 95865 rows, 0.010 GB
    create_visibility: flagged 0/95865 visibilities below elevation limit 0.261799 (rad)



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_19_4.png


.. parsed-literal::

    Processing context timeslice
    create_visibility: 95865 rows, 0.010 GB
    create_visibility: flagged 0/95865 visibilities below elevation limit 0.261799 (rad)



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_19_6.png


.. parsed-literal::

    Processing context wstack
    create_visibility: 95865 rows, 0.010 GB
    create_visibility: flagged 0/95865 visibilities below elevation limit 0.261799 (rad)



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_19_8.png


.. parsed-literal::

    Processing context wprojection
    create_visibility: 95865 rows, 0.010 GB
    create_visibility: flagged 0/95865 visibilities below elevation limit 0.261799 (rad)



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_19_10.png


Make the image using the different approaches. We will evaluate the
results using a number of plots:

-  The error in fitted versus the radius. The ideal result is a
   straightline fitted: flux = DFT flux
-  The offset in RA versus the offset in DEC. The ideal result is a
   cluster around 0 pixels.

The sampling in w is set to provide 2% decorrelation at the half power
point of the primary beam.

.. code:: ipython3

    contexts = ['2d', 'facets', 'timeslice', 'wstack', 'wprojection']
    
    for context in contexts:
    
        targetimage_list = [create_image_from_visibility(vt, npixel=npixel, cellsize=0.001, nchan=1,
                                                   polarisation_frame=PolarisationFrame('stokesI'))]
        
        vt_list = [vt]
    
    
        print('Processing context %s' % context)
        if context == 'wprojection':
            future = invert_list_arlexecute_workflow(vt_list, targetimage_list, context='2d', gcfcf=[gcfcf_clipped])
        
        elif context == 'facets':
            future = invert_list_arlexecute_workflow(vt_list, targetimage_list, context=context, facets=8)
            
        elif context == 'timeslice':
            future = invert_list_arlexecute_workflow(vt_list, targetimage_list, context=context, vis_slices=vis_timeslices(vt, 'auto'))
    
        elif context == 'wstack':
            future = invert_list_arlexecute_workflow(vt_list, targetimage_list, context=context, vis_slices=31)
    
        else:
            future = invert_list_arlexecute_workflow(vt_list, targetimage_list, context=context)
            
        result = arlexecute.compute(future, sync=True)
        targetimage = result[0][0]
    
        show_image(targetimage)
        plt.title(context)
        plt.show()
    
        print("Dirty Image %s" % qa_image(targetimage, context="imaging-fits notebook, using processor %s" % context))
    
        export_image_to_fits(targetimage, '%s/imaging-fits_dirty_%s.fits' % (results_dir, context))
        comps = find_skycomponents(targetimage, fwhm=1.0, threshold=10.0, npixels=5)
    
        plt.clf()
        for comp in comps:
            distance = comp.direction.separation(model.phasecentre)
            dft_flux = sum_visibility(vt, comp.direction)[0]
            err = (comp.flux[0, 0] - dft_flux) / dft_flux
            plt.plot(distance, err, '.', color='r')
        plt.ylabel('Fractional error of image vs DFT')
        plt.xlabel('Distance from phasecentre (deg)')
        plt.title(
            "Fractional error in %s recovered flux vs distance from phasecentre" %
            context)
        plt.show()
    
        checkpositions = True
        if checkpositions:
            plt.clf()
            for i in range(len(comps)):
                ocomp, sep = find_nearest_skycomponent(comps[i].direction, original_comps)
                plt.plot(
                    (comps[i].direction.ra.value - ocomp.direction.ra.value) /
                    targetimage.wcs.wcs.cdelt[0],
                    (comps[i].direction.dec.value - ocomp.direction.dec.value) /
                    targetimage.wcs.wcs.cdelt[1],
                    '.',
                    color='r')
    
            plt.xlabel('delta RA (pixels)')
            plt.ylabel('delta DEC (pixels)')
            plt.title("%s: Position offsets" % context)
            plt.show()


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 999999.99999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 256, 256]
    Processing context 2d



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_1.png


.. parsed-literal::

    Dirty Image Quality assessment:
    	Origin: qa_image
    	Context: imaging-fits notebook, using processor 2d
    	Data:
    		shape: '(1, 1, 256, 256)'
    		max: '108.2052676771443'
    		min: '-7.218746872709628'
    		maxabs: '108.2052676771443'
    		rms: '4.886194600492366'
    		sum: '1981.2706452319446'
    		medianabs: '1.1714737857311381'
    		medianabsdevmedian: '1.0443013200662428'
    		median: '-0.6052575706060335'
    
    find_skycomponents: Finding components in Image by segmentation
    find_skycomponents: Identified 28 segments



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_3.png



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_4.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 999999.99999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 256, 256]
    Processing context facets



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_6.png


.. parsed-literal::

    Dirty Image Quality assessment:
    	Origin: qa_image
    	Context: imaging-fits notebook, using processor facets
    	Data:
    		shape: '(1, 1, 256, 256)'
    		max: '176.5281834793862'
    		min: '-14.780837361445153'
    		maxabs: '176.5281834793862'
    		rms: '6.628584747730007'
    		sum: '-1283.5795631501499'
    		medianabs: '1.184416535225445'
    		medianabsdevmedian: '1.0709711877970065'
    		median: '-0.6876741395738546'
    
    find_skycomponents: Finding components in Image by segmentation
    find_skycomponents: Identified 36 segments



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_8.png



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_9.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 999999.99999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 256, 256]
    Processing context timeslice



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_11.png


.. parsed-literal::

    Dirty Image Quality assessment:
    	Origin: qa_image
    	Context: imaging-fits notebook, using processor timeslice
    	Data:
    		shape: '(1, 1, 256, 256)'
    		max: '172.81835860062475'
    		min: '-5.869581605876343'
    		maxabs: '172.81835860062475'
    		rms: '6.478730515924426'
    		sum: '3375.3494713450837'
    		medianabs: '1.123216489240447'
    		medianabsdevmedian: '1.0059823897153521'
    		median: '-0.642221415360976'
    
    find_skycomponents: Finding components in Image by segmentation
    find_skycomponents: Identified 36 segments



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_13.png



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_14.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 999999.99999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 256, 256]
    Processing context wstack



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_16.png


.. parsed-literal::

    Dirty Image Quality assessment:
    	Origin: qa_image
    	Context: imaging-fits notebook, using processor wstack
    	Data:
    		shape: '(1, 1, 256, 256)'
    		max: '6788.299487523429'
    		min: '-6546.887366891919'
    		maxabs: '6788.299487523429'
    		rms: '1511.5287235044607'
    		sum: '-39642.3453714347'
    		medianabs: '1010.5932765878683'
    		medianabsdevmedian: '1010.1380673274541'
    		median: '-2.0286270287858574'
    
    find_skycomponents: Finding components in Image by segmentation
    find_skycomponents: Identified 28 segments



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_18.png



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_19.png


.. parsed-literal::

    create_image_from_visibility: Parsing parameters to get definition of WCS
    create_image_from_visibility: Defining single channel Image at <SkyCoord (ICRS): (ra, dec) in deg
        (15., -45.)>, starting frequency 100000000.0 Hz, and bandwidth 999999.99999 Hz
    create_image_from_visibility: uvmax = 262.634709 wavelengths
    create_image_from_visibility: Critical cellsize = 0.001904 radians, 0.109079 degrees
    create_image_from_visibility: Cellsize          = 0.001 radians, 0.0572958 degrees
    create_image_from_visibility: image shape is [1, 1, 256, 256]
    Processing context wprojection



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_21.png


.. parsed-literal::

    Dirty Image Quality assessment:
    	Origin: qa_image
    	Context: imaging-fits notebook, using processor wprojection
    	Data:
    		shape: '(1, 1, 256, 256)'
    		max: '169.93958570049563'
    		min: '-5.924830370028082'
    		maxabs: '169.93958570049563'
    		rms: '6.464558076006836'
    		sum: '2837.406839134143'
    		medianabs: '1.13144273176894'
    		medianabsdevmedian: '1.0134461773467502'
    		median: '-0.6544929780631125'
    
    find_skycomponents: Finding components in Image by segmentation
    find_skycomponents: Identified 36 segments



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_23.png



.. image:: imaging-fits_arlexecute_files/imaging-fits_arlexecute_21_24.png


