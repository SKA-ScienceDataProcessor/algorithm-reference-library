.. Fourier processing

Fourier processing
******************

There are many algorithms for imaging, using different approaches to correct for various effects:

+ Simple 2D transforms
+ Partitioned image (i.e. faceted) and uv transforms
+ W projection
+ W snapshots
+ W slices
+ A projection variants
+ Visibility coalescence and de-coalescence
+ MFS variants
+ Differential residuals calculations

Since the scale of SDP is so much larger than previous telescopes, it is not clear which scaling strategies and
algorithms are going to offer the best performance. For this reason, it is important the synthesis framework not be
restrictive.

All the above functions are linear in the visibilities and image. The 2D transform is correct for sufficiently
restricted context. Hence we will layer all algorithms on top of the 2D transform. This means that a suitable
framework decomposes the overall transform into suitable linear combinations of invocations of 2D transforms. We can
use python iterators to perform the subsectioning. For example, the principal image iteration via a raster
implemented by a python generator::

        m31model=create_test_image()
        for ipatch in raster(m31model, nraster=2):
            # each image patch can be used to add to the visibility data
            vis + = predict_2d(vis, ipatch, params)

        # For image partitioning and snapshot processing
        iraster, interval = find_optimum_iraster_times(vis, model)
        m31model=create_test_image()
        for ipatch in raster(m31model, nraster=iraster):
            for subvis in snapshot(vis, interval=interval):
                # each patch can be used to add to the visibility data
                subvis + = predict_2d(subvis, ipatch, params)

This relies upon the data objects (model and vis) possessing sufficient meta data to enable operations such as phase
rotation from one frame to another.

In addition, iteration through the visibility data must tbe varied:

+ By time
+ By frequency
+ By w
+ By parallactic angle

The Visibility API should support these forms of iteration.

The pattern used in these algorithms is abstracted in the following diagram:

.. image:: ./ARL_fourier_processing.png
      :width: 1024px

These can be defined as stateless functions::

    def predict_image_partition(vis, model, predict_function, params):
        """ Predict using image partitions

        """
        nraster = get_parameter(params, "image_partitions", 3)
        for ipatch in raster(model, nraster=nraster):
            predict_function(vis, ipatch, params)

        return vis


    def predict_wslices(vis, model, predict_function, params):
        """ Predict using image partitions

        """
        wstep = get_parameter(params, "wstep", 1000)
        for ipatch in wslice(model, wstep):
            predict_function(vis, ipatch, params)

        return vis

These can be nested as such::

    predict_wslices(vis, model, predict_function=predict_image_partition)

This will perform wslice transforms and inside those, image partition transforms.

Parallel processing
*******************

ARL uses parallel processing to speed up some calculations. It is not intended to indicate a preference for how
parallel processing should be implemented in SDP.

We use an openMP-like package `pypm <https://github.com/classner/pymp/>`_. An example is to be found in
arl/fourier_transforms/invert-timeslice. The data are divided into timeslices and then processed in parallel::

   def invert_timeslice(vis, im, dopsf=False, **kwargs):
       """ Invert using time slices (top level function)

       Use the image im as a template. Do PSF in a separate call.

       :param vis: Visibility to be inverted
       :param im: image template (not changed)
       :param dopsf: Make the psf instead of the dirty image
       :param nprocessor: Number of processors to be used (1)
       :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

       """
       log.debug("invert_timeslice: inverting using time slices")
       resultimage = create_image_from_array(im.data, im.wcs)
       resultimage.data = pymp.shared.array(resultimage.data.shape)
       resultimage.data *= 0.0

       nproc = get_parameter(kwargs, "nprocessor", 1)

       nchan, npol, _, _ = im.data.shape

       totalwt = numpy.zeros([nchan, npol], dtype='float')

       if nproc > 1:
           # We need to tell pymp that some arrays are shared
           resultimage.data = pymp.shared.array(resultimage.data.shape)
           resultimage.data *= 0.0
           totalwt = pymp.shared.array([nchan, npol])

           # Extract the slices and run invert_timeslice_single on each one in parallel
           nslices = 0
           rowses = []
           for rows in vis_timeslice_iter(vis, **kwargs):
               nslices += 1
               rowses.append(rows)

           log.debug("invert_timeslice: Processing %d time slices %d-way parallel" % (nslices, nproc))
           with pymp.Parallel(nproc) as p:
               for index in p.range(0, nslices):
                   visslice = create_visibility_from_rows(vis, rowses[index])
                   workimage, sumwt = invert_timeslice_single(visslice, im, dopsf, **kwargs)
                   resultimage.data += workimage.data
                   totalwt += sumwt

       else:
           # Do each slice in turn
           for rows in vis_timeslice_iter(vis, **kwargs):
               visslice=create_visibility_from_rows(vis, rows)
               workimage, sumwt = invert_timeslice_single(visslice, im, dopsf, **kwargs)
               resultimage.data += workimage.data
               totalwt += sumwt

       return resultimage, totalwt

