
Simple demonstration of the use of generic dask functions, as wrapped in arlexecute.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import os
    import sys
    
    sys.path.append(os.path.join('..','..'))
    
    import numpy
    
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    
    from data_models.polarisation import PolarisationFrame
    
    from processing_components.component_support.arlexecute import arlexecute
    from processing_components.component_support.generic_components import generic_image_component
    from processing_components.util.testing_support import create_test_image
    
    import logging
    
    logging.basicConfig(filename='simple-dask.log',
                                filemode='a',
                                format='%(thread)s %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)

Set up arlexecute to use dask

.. code:: ipython3

    arlexecute.set_client(use_dask=True)

.. code:: ipython3

    frequency = numpy.array([1e8])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    model_graph = arlexecute.execute(create_test_image)(frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                                             polarisation_frame=PolarisationFrame('stokesI'))
    model = arlexecute.compute(model_graph, sync=True)


.. parsed-literal::

    arlexecute.compute: Synchronous execution took 1.169 seconds


.. parsed-literal::

    WARNING: FITSFixedWarning: The WCS transformation has more axes (4) than the image it is associated with (0) [astropy.wcs.wcs]


Define a simple function to take the square root of an image

.. code:: ipython3

    def imagerooter(im, **kwargs):
        im.data = numpy.sqrt(numpy.abs(im.data))
        return im
    
    root = generic_image_component(imagerooter, model_graph, facets=4)

Run the graph directly

.. code:: ipython3

    result = arlexecute.compute(root, sync=True)
    numpy.testing.assert_array_almost_equal_nulp(result.data**2, numpy.abs(model.data), 7)


.. parsed-literal::

    arlexecute.compute: Synchronous execution took 0.157 seconds


.. parsed-literal::

    WARNING: FITSFixedWarning: The WCS transformation has more axes (4) than the image it is associated with (0) [astropy.wcs.wcs]


.. code:: ipython3

    arlexecute.close()
