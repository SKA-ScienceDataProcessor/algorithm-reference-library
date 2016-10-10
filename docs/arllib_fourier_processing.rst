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
+ Partitioned image (i.e. faceted) and uv transforms
+ W projection
+ W snapshots
+ W slices
+ A projection variants
+ MFS variants

Approach
--------

All the above functions are linear in the visibilities and image. The 2D transform is correct for sufficiently
restricted context. Hence we will layer all algorithms on top of the 2D transform. This means that a suitable
framework decomposes the overall transform into suitable linear combinations of invocations of 2D transforms. We can
use python iterators to perform the subsectioning. The subimage information such as size and coordinates is passed
using an Image.

The interface is therefore::

    class vis_iterator_base():
        def __init__(vis, params):
            """Initialise the iterator
            """
            self.maximum_index = 10
        self.index = 0

        def __iter__(self):
            """ Return the iterator itself
            """
            return self

        def __next__(self):
            try:
                result = self.vis.select(self.index)
            except IndexError:
                raise StopIteration
            self.index += 1
            return result

    class model_iterator_base():
        def __init__(model, params):
            """Initialise the iterator for prediction
            """

            self.maximum_index = 10
            self.index = 0

        def __iter__(self):
            """ Return the iterator itself
            """
            return self

        def __next__(self):
            try:
                result = self.vis.select(self.index)
            except IndexError:
                raise StopIteration
            self.index += 1
            return result



The pattern used in these algorithms is abstracted in the following diagram:

.. image:: ./ARL_fourier_processing.png
      :width: 1024px

We express this pattern in python using generators::

    def predict_ip(model, vis, params):
        # Define and use a predict engine for image partitions
        for ftip in ftpredict_image_partition(model, vt, params):
            # Define 2D predict engine for this image partition
            for ft2d in ftpredict_2d(ftip.extract_model(), ftip.extract_vis(), ftip.params):
                # Use the 2D predict function and insert the answer back into the image partition predict engine
                # The image partition predict generator knows where to put it
                ftip.insert_visibility(ft2d.predict())

     def invert_ip_2d(model, vis, params):
       for ftip in ftinvert_image_partition(model, vt, params):
            # Define an 2D invert engine and then use the predict function
           for ft2d in ftinvert_2d(ftip.extract_model(), ftip.extract_vis(), ftip.params):
                # Use the 2D invert engine to insert the answer back into the correct place in the whole image
                ftip.insert_images(ftinvert_2d(ftip.extract_model(), ftip.extract_vis(), ftip.params).invert())

    def predict_fp_ip_2d(model, vis, params):
        # Define and use a predict engine for fourier partitions
        for ftfp in ftpredict_fourier_partition(model, vt, params)
            # Define image partition predict engine for this fourier partition
            for ftip in ftpredict_image_partition(ftfp.extract_model(), ftfp.extract_vis(), ftfp.params):
                # Define 2D predict engine for this image partition
                for ft2d in ftpredict_2d(ftip.extract_model(), ftip.extract_vis(), ftip.params):
                    # Use the 2D predict function and insert the answer back into the image partition predict engine
                    # The image partition predict generator knows where to put it
                    ftip.insert_visibility(ft2d.predict())
                ftfp.insert_visibility(ftip.predict())


