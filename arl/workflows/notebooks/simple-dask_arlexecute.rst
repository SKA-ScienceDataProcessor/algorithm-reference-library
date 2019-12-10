Simple demonstration of the use of arlexecute
=============================================

.. code:: ipython3

    import os
    import sys
    
    sys.path.append(os.path.join('..','..'))
    
    import numpy
    
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    
    from arl.data_models import PolarisationFrame
    
    from arl.processing_components import image_scatter_facets, image_gather_facets, create_test_image
    from arl.wrappers.arlexecute.execution_support.arlexecute import arlexecute
    
    import logging
    
    logging.basicConfig(filename='simple-dask.log',
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
    
    mpl_logger = logging.getLogger("matplotlib") 
    mpl_logger.setLevel(logging.WARNING) 

Set up arlexecute to use dask

.. code:: ipython3

    arlexecute.set_client(use_dask=True)

.. code:: ipython3

    frequency = numpy.array([1e8])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    model_graph = arlexecute.execute(create_test_image)(frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                                             polarisation_frame=PolarisationFrame('stokesI'))
    model = arlexecute.compute(model_graph, sync=True)

Define a simple function to take the square root of an image

.. code:: ipython3

    def imagerooter(im, **kwargs):
        im.data = numpy.sqrt(numpy.abs(im.data))
        return im
    
    facets_graph = arlexecute.execute(image_scatter_facets, nout=16)(model_graph, facets=4)
    root_graph = [arlexecute.execute(imagerooter)(facet) for facet in facets_graph]
    gathered = arlexecute.execute(image_gather_facets)(root_graph, model_graph, facets=4)

Run the graph directly

.. code:: ipython3

    result = arlexecute.compute(gathered, sync=True)
    numpy.testing.assert_array_almost_equal_nulp(result.data**2, numpy.abs(model.data), 7)

.. code:: ipython3

    arlexecute.close()
