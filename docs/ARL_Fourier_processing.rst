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

In addition, iteration through the visibility data can be on various orders::

+ By time
+ By frequency
+ By w

The Visibility API should support these forms of iteration.

The pattern used in these algorithms is abstracted in the following diagram:

.. image:: ./ARL_fourier_processing.png
      :width: 1024px

The layering of predict and invert classes is shown below:

.. image:: ARL_predict_layering.png
      :width: 1024px

.. image:: ARL_invert_layering.png
      :width: 1024px

The top level functions are in green. All capability is therefore layered on two functions, predict_2d and invert_2d.

Not all combinations currently work. The following are supported:

+ W Projection: {predict | invert}_wprojection
+ W Stacking:  {predict | invert}_wstack
+ Image plane facets:  {predict | invert}_wfacets
+ W Stacking + W Projection: {predict | invert}_wstack_wprojection
+ W Stacking + Image plane facets:

