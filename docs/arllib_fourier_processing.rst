.. ARL documentation master file

.. toctree::
   :name: mastertoc
   :maxdepth: 2


:index:`Fourier processing`
***************************


Goals
-----

There are many algorithms for imaging, using different approaches to correct for various effects:

+ Simple 2D transforms
+ Faceted transforms
+ W projection
+ W snapshots
+ W slices
+ MFS variants of all the above

Approach
--------

Expressing any one of these is straightforward but combinations can become messy unless a suitable abstraction is found.

The pattern used in these algorithms is abstracted in the following diagram:

.. image:: ./ARL_fourier_processing.png
      :width: 1024px

Thus we nest imagers. For imaging with w faceting::

    ft_facet = fouriertransform_faceting(vt, sm, params) #
    with f = ft_facet.next():
        f.insert_images(invert_visibility(f.vis, f.sm, f.params))

    # The function fouriertransform_imaging has the interface::
    f.next()    # Get the next chunk
    f.vis       # Attribute holding the visibility
    f.sm        # Attribute holding the sky model
    f.params    # Attribute holding the parameters
    f.insert_images
    f.extract_images
    f.insert_visibilities
    f.extract_visibilities

