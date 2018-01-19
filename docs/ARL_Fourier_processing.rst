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

Since the scale of SKA is so much larger than previous telescopes, it is not clear which scaling strategies and
algorithms are going to offer the best performance. For this reason, it is important the synthesis framework not be
restrictive.

All the above functions are linear in the visibilities and image. The 2D transform is correct for sufficiently
restricted context. Hence we layer all algorithms on top of the 2D transform. This means that a suitable
framework decomposes the overall transform into suitable linear combinations of invocations of 2D transforms.

ARL has two ways of invoking the calibration and imaging capabilities: python functions and Dask delayed functions.


In the first approach we use python iterators to perform the
subsectioning. For example, the principal image iteration
via a raster implemented by a python generator. The styel of this approach is::

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
rotation from one frame to another. See e.g. :py:mod:`arl.imaging.imaging_context.invert_function`

The second approach is based on the same underlying functions, predict_2d_base and invert_2d_base, but uses lazy
evaluation implemented by the Dask.delayed package. See e.g. :py:mod:`arl.graphs.delayed.create_invert_graph`

The Visibility API supports these forms of iteration.

To enable efficient graph processing, the units of processing are kept small. Each should be doable in a few minutes.




