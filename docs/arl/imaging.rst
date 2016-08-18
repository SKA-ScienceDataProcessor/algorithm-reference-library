
This script works through the high level arl interface to crocodile, making a fake data set and then deconvolving it. Finally the full and residual visibility are plotted.
===========================================================================================================================================================================

.. code:: python

    import sys, os
    sys.path.append('../..') 
    print(sys.path)
    print(os.getcwd())
    
    from matplotlib import pylab
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    pylab.rcParams['image.cmap'] = 'rainbow'
    
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
    from astropy import units as u
    
    from matplotlib import pyplot as plt
    
    from arl.deconvolve_image import deconvolve_cube
    from arl.define_visibility import create_visibility
    from arl.fourier_transform import *
    from arl.define_skymodel import create_skymodel_from_image, add_component_to_skymodel, create_skycomponent, find_point_source
    from arl.define_image import show_image, create_image_from_fits, save_image_to_fits, replicate_image
    from arl.simulate_visibility import filter_configuration, create_named_configuration


.. parsed-literal::

    ['', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python35.zip', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/plat-darwin', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/lib-dynload', '/Users/timcornwell/env/lib/python3.5/site-packages', '/Users/timcornwell/env/lib/python3.5/site-packages/IPython/extensions', '/Users/timcornwell/.ipython', '../..']
    /Users/timcornwell/Code/crocodile/examples/arl


.. code:: python

    # We construct a VLA configuration and then shrink it to match our test image.
    
    kwargs = {}
    
    vlaa = filter_configuration(create_named_configuration('VLAA'), **kwargs)
    vlaa.data['xyz']=vlaa.data['xyz']/10.0
    
    
    # We create the visibility. This just makes the uvw, time, antenna1, antenna2, weight columns in a table
    
    times = numpy.arange(-numpy.pi/2.0, +numpy.pi/2.0,0.05)
    frequency = numpy.array([1e8])
    
    reffrequency = numpy.max(frequency)
    phasecentre = SkyCoord(0.0*u.rad, u.rad*numpy.pi/4, frame='icrs', equinox=2000.0)
    vt = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)
    
    
    # Plot the synthesized uv coverage, including for MFS
    
    plt.clf()
    for f in frequency:
        x=f/const.c
        plt.plot(x*vt.data['uvw'][:,0], x*vt.data['uvw'][:,1], '.', color='b')
        plt.plot(-x*vt.data['uvw'][:,0], -x*vt.data['uvw'][:,1], '.', color='r')


.. parsed-literal::

    filter_configuration: No filter implemented yet
    visibility.create_visibility: Created 22113 rows


.. code:: python

    # Read the venerable test image, constructing an image
    
    m31image = create_image_from_fits("./data/models/M31.MOD")
    fig = plt.figure()
    cellsize=180.0*0.0001/numpy.pi
    m31image.wcs.wcs.cdelt[0]=-cellsize
    m31image.wcs.wcs.cdelt[1]=+cellsize
    m31image.wcs.wcs.radesys='ICRS'
    m31image.wcs.wcs.equinox=2000.00
    
    # Show the model image
    fig.add_subplot(111, projection=m31image.wcs)
    plt.imshow(m31image.data, origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.show()
    
    # This image is only 2 dimensional. We need extra axes frequency and stokes.
    
    m31image4D=replicate_image(m31image, shape=[1, 1, 4, len(frequency)])
    m31sm = create_skymodel_from_image(m31image4D)
    
    # We need a linear reference frame to inset a model source. This is a bit involved die to the Astropy way of doing
    # things
    wall = m31image.wcs
    wall.wcs.radesys='ICRS'
    wall.wcs.equinox=2000.00
    print(wall.wcs.radesys)
    print(wall.wcs.equinox)
    sc=pixel_to_skycoord(128, 128, wall, 1, 'wcs')
    compabsdirection=SkyCoord("-1.0d", "37.0d", frame='icrs', equinox=2000.0)
    pixloc = skycoord_to_pixel(compabsdirection, wall, 1)
    scrt = pixel_to_skycoord(pixloc[0], pixloc[1], wall, 1, 'wcs')
    sof=sc.skyoffset_frame()
    compreldirection = compabsdirection.transform_to(sof)
    
    # Create a skycomponent and add it to the skymodel
    comp1= create_skycomponent(flux=numpy.array([[1.0, 0.0, 0.0, 0.0]]), frequency=frequency, direction=compreldirection)
    m31sm=add_component_to_skymodel(m31sm, comp1)


.. parsed-literal::

    create_image_from_fits: Max, min in /Users/timcornwell/Code/crocodile//./data/models/M31.MOD = 1.006458, 0.000000
    replicate_image: replicating shape (256, 256) to (1, 4, 256, 256)
    ICRS
    2000.0


.. code:: python

    # Now we can predict_visibility the visibility from this skymodel
    kwargs={'wstep':100.0, 'npixel':256, 'cellsize':0.0001}
    vt = predict_visibility(vt, m31sm, **kwargs)
    
    # To check that we got the prediction right, plot the amplitude of the visibility.
    uvdist=numpy.sqrt(vt.data['uvw'][:,0]**2+vt.data['uvw'][:,1]**2)
    plt.clf()
    plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]), '.')
    plt.xlabel('uvdist')
    plt.ylabel('Amp Visibility')
    plt.show()


.. parsed-literal::

    imaging.create_wcs_from_visibility: Parsing kwargs to get definition of WCS
    imaging.create_wcs_from_visibility: Defining Image at <SkyCoord (ICRS): (ra, dec) in deg
        (0.0, 45.0)>, frequency 100000000.0 Hz, and bandwidth 100000000.0 Hz
    imaging.create_wcs_from_visibility: uvmax = 1533.754509 lambda
    imaging.create_wcs_from_visibility: Critical cellsize = 0.000326 radians, 0.018678 degrees
    imaging.create_wcs_from_visibility: Cellsize          = 0.000100 radians, 0.005730 degrees
    imaging.predict_visibility: Predicting Visibility from sky model images
    imaging.predict_visibility: Image cellsize 0.000100 radians
    imaging.predict_visibility: Field of view 0.025600 radians
    imaging.predict_visibility: Making w-kernel cache of 12 kernels
    imaging.predict_visibility: Predicting from image channel 0, polarisation 0
    imaging.predict_visibility: Predicting from image channel 0, polarisation 1
    imaging.predict_visibility: Predicting from image channel 0, polarisation 2
    imaging.predict_visibility: Predicting from image channel 0, polarisation 3
    imaging.predict_visibility: Finished predicting Visibility from sky model images
    imaging.predict_visibility: Predicting Visibility from sky model components
    imaging.predict_visibility: Cartesian representation of component 0 = (0.999291, -0.013938, 0.034969)
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 0
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 1
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 2
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 3
    imaging.predict_visibility: Finished predicting Visibility from sky model components


.. code:: python

    # Make the dirty image and point spread function
    
    kwargs={}
    kwargs['npixel']=512
    kwargs['cellsize']=0.0001
    kwargs['wstep']=30.0
    dirty, psf, sumwt = invert_visibility(vt, **kwargs)
    show_image(dirty)
    print("Max, min in dirty image = %.6f, %.6f, sum of weights = %f" % (dirty.data.max(), dirty.data.min(), sumwt))
    
    print("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" % (psf.data.max(), psf.data.min(), sumwt))
    
    save_image_to_fits(dirty, 'dirty.fits')
    save_image_to_fits(psf, 'psf.fits')
    m31compnew = find_point_source(dirty)
    
    
    # Deconvolve using clean
    
    kwargs={'niter':100, 'threshold':0.001, 'fracthresh':0.01}
    comp, residual = deconvolve_cube(dirty, psf, **kwargs)
    
    # Show the results
    
    fig=show_image(comp)
    fig=show_image(residual)


.. parsed-literal::

    imaging.invert_visibility: Inverting Visibility to make dirty and psf
    imaging.create_wcs_from_visibility: Parsing kwargs to get definition of WCS
    imaging.create_wcs_from_visibility: Defining Image at <SkyCoord (ICRS): (ra, dec) in deg
        (0.0, 45.0)>, frequency 100000000.0 Hz, and bandwidth 100000000.0 Hz
    imaging.create_wcs_from_visibility: uvmax = 1533.754509 lambda
    imaging.create_wcs_from_visibility: Critical cellsize = 0.000326 radians, 0.018678 degrees
    imaging.create_wcs_from_visibility: Cellsize          = 0.000100 radians, 0.005730 degrees
    imaging.invert_visibility: Specified npixel=512, cellsize = 0.000100 rad, FOV = 0.051200 rad
    imaging.invert_visibility: Making w-kernel cache of 39 kernels
    imaging.invert_visibility: Inverting channel 0, polarisation 0
    imaging.invert_visibility: Inverting channel 0, polarisation 1
    imaging.invert_visibility: Inverting channel 0, polarisation 2
    imaging.invert_visibility: Inverting channel 0, polarisation 3
    imaging.invert_visibility: Finished making dirty and psf
    [[-0.02325099 -0.04737639 -0.09495533 ..., -0.12593225 -0.07047448
      -0.03191264]
     [-0.07621128 -0.0788952  -0.1054796  ..., -0.18039444 -0.13957662
      -0.09956341]
     [-0.14003592 -0.11952093 -0.11519131 ..., -0.22324736 -0.20560679
      -0.17290562]
     ..., 
     [ 0.03668866 -0.01625759 -0.06548818 ...,  0.02411814  0.0677868
       0.07053067]
     [ 0.03712874 -0.01213939 -0.067555   ..., -0.01364075  0.04192436
       0.05929281]
     [ 0.01557495 -0.02447878 -0.08098756 ..., -0.06723331 -0.00708045
       0.02306276]]
    Max, min in dirty image = 20.834159, -1.187682, sum of weights = 0.038254
    Max, min in PSF         = 1.000000, -0.111255, sum of weights = 0.038254
    imaging.point_source_find: Finding components in Image
    imaging.point_source_find: Found peak at pixel coordinates [  0   2 281 248]
    imaging.point_source_find: Found peak at world coordinates <SkyCoord (ICRS): (ra, dec) in deg
        (0.06498543, 45.14322122)>
    imaging.point_source_find: Flux is [[ 20.82519145  20.82585086  20.83415936  20.82585086]]
    clean.clean: Processing pol 0, channel 0
    Peak of PSF = 1.0 at (256, 256)
    Peak of Dirty = 20.8251914472 at (281, 248)
    Coupling matrix =
     [[ 1.          0.98254714  0.76027592  0.19824986]
     [ 0.98254714  0.96555194  0.74886582  0.19720193]
     [ 0.76027592  0.74886582  0.60082915  0.18228196]
     [ 0.19824986  0.19720193  0.18228196  0.10628513]]
    Max abs in dirty Image = 20.825191
    Start of minor cycle
    This minor cycle will stop at 100 iterations or peak < 0.208251914472
    Minor cycle 0, peak [ 17.18728983  17.12492822  16.27034915  12.36866438] at [268, 260, 3]
    Minor cycle 10, peak [ 7.63029298  7.59835038  7.16228706  5.47100467] at [326, 267, 3]
    Minor cycle 20, peak [ 4.3166189   4.29489324  4.0014179   2.90478914] at [279, 218, 3]
    Minor cycle 30, peak [ 4.30240934  4.2778072   3.91617755  1.97903513] at [247, 185, 3]
    Minor cycle 40, peak [ 3.57242166  3.53376612  3.03722138  1.61348499] at [291, 240, 3]
    Minor cycle 50, peak [ 2.36798999  2.35098999  2.12138038  1.24056276] at [247, 299, 3]
    Minor cycle 60, peak [ 1.30795801  1.30260644  1.23137134  0.91782787] at [328, 281, 3]
    Minor cycle 70, peak [ 1.38428264  1.37637004  1.2638341   0.75907465] at [231, 172, 3]
    Minor cycle 80, peak [ 1.2516061   1.24305305  1.12527632  0.62508662] at [340, 274, 3]
    Minor cycle 90, peak [ 4.20912436  4.12986082  3.10319254  0.46156967] at [292, 290, 2]
    End of minor cycles
    clean.clean: Skipping pol 1, channel 0
    clean.clean: Skipping pol 2, channel 0
    clean.clean: Skipping pol 3, channel 0
    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ..., 
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]]
    [[-0.06717577 -0.09190788 -0.14082887 ..., -0.11467402 -0.0647041
      -0.03298608]
     [-0.11743105 -0.12038271 -0.14816014 ..., -0.16336482 -0.12745767
      -0.09362189]
     [-0.17839381 -0.15808922 -0.15506236 ..., -0.20300025 -0.18938194
      -0.16181118]
     ..., 
     [-0.09527682 -0.16713035 -0.19011937 ..., -0.1456796  -0.14420009
      -0.12340903]
     [-0.09832261 -0.17576047 -0.21863223 ..., -0.16266263 -0.15017232
      -0.11226161]
     [-0.11811895 -0.19437072 -0.24957486 ..., -0.16073699 -0.14346741
      -0.09198905]]


.. code:: python

    # Predict the visibility of the model
    
    kwargs={'wstep':30.0}
    vt = predict_visibility(vt, m31sm, **kwargs)
    modelsm=create_skymodel_from_image(comp)
    vtmodel = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)
    vtmodel.data = vt.data.copy()
    vtmodel=predict_visibility(vtmodel, modelsm, **kwargs)
    
    
    # Now we will plot the original visibility and the residual visibility.
    
    uvdist=numpy.sqrt(vt.data['uvw'][:,0]**2+vt.data['uvw'][:,1]**2)
    plt.clf()
    plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]), '.', color='b')
    plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]-vtmodel.data['vis'][:,0,0]), '.', color='r')
    plt.xlabel('uvdist')
    plt.ylabel('Amp Visibility')
    plt.show()


.. parsed-literal::

    imaging.create_wcs_from_visibility: Parsing kwargs to get definition of WCS
    imaging.create_wcs_from_visibility: Defining Image at <SkyCoord (ICRS): (ra, dec) in deg
        (0.0, 45.0)>, frequency 100000000.0 Hz, and bandwidth 100000000.0 Hz
    imaging.create_wcs_from_visibility: uvmax = 1533.754509 lambda
    imaging.create_wcs_from_visibility: Critical cellsize = 0.000326 radians, 0.018678 degrees
    imaging.create_wcs_from_visibility: Cellsize          = 0.000163 radians, 0.009339 degrees
    imaging.predict_visibility: Predicting Visibility from sky model images
    imaging.predict_visibility: Image cellsize 0.000100 radians
    imaging.predict_visibility: Field of view 0.025600 radians
    imaging.predict_visibility: Making w-kernel cache of 39 kernels
    imaging.predict_visibility: Predicting from image channel 0, polarisation 0
    imaging.predict_visibility: Predicting from image channel 0, polarisation 1
    imaging.predict_visibility: Predicting from image channel 0, polarisation 2
    imaging.predict_visibility: Predicting from image channel 0, polarisation 3
    imaging.predict_visibility: Finished predicting Visibility from sky model images
    imaging.predict_visibility: Predicting Visibility from sky model components
    imaging.predict_visibility: Cartesian representation of component 0 = (0.999291, -0.013938, 0.034969)
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 0
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 1
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 2
    imaging.predict_visibility: Predicting from component 0 channel 0, polarisation 3
    imaging.predict_visibility: Finished predicting Visibility from sky model components
    visibility.create_visibility: Created 22113 rows
    imaging.create_wcs_from_visibility: Parsing kwargs to get definition of WCS
    imaging.create_wcs_from_visibility: Defining Image at <SkyCoord (ICRS): (ra, dec) in deg
        (0.0, 45.0)>, frequency 100000000.0 Hz, and bandwidth 100000000.0 Hz
    imaging.create_wcs_from_visibility: uvmax = 1533.754509 lambda
    imaging.create_wcs_from_visibility: Critical cellsize = 0.000326 radians, 0.018678 degrees
    imaging.create_wcs_from_visibility: Cellsize          = 0.000163 radians, 0.009339 degrees
    imaging.predict_visibility: Predicting Visibility from sky model images
    imaging.predict_visibility: Image cellsize 0.000100 radians
    imaging.predict_visibility: Field of view 0.051200 radians
    imaging.predict_visibility: Making w-kernel cache of 39 kernels
    imaging.predict_visibility: Predicting from image channel 0, polarisation 0
    imaging.predict_visibility: Predicting from image channel 0, polarisation 1
    imaging.predict_visibility: Predicting from image channel 0, polarisation 2
    imaging.predict_visibility: Predicting from image channel 0, polarisation 3
    imaging.predict_visibility: Finished predicting Visibility from sky model images


