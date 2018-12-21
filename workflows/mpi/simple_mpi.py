"""Simple demonstration of the use of ARL functions with MPI

Run with:

mpiexec -n 4 python simple_mpi.py

"""


import logging

import numpy
from mpi4py import MPI

import astropy.units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from processing_library.image.operations import create_empty_image_like
from processing_components.image.operations import qa_image
from processing_components.simulation.testing_support import create_test_image
from processing_components.image.gather_scatter import image_gather_facets, image_scatter_facets

# Uncomment following line if profiling with extrae/paraver toolset
#import pyextrae.mpi as pyextrae

#from matplotlib import pyplot as plt

# Define a simple function to take the square root of an image
def imagerooter(image_list) -> list():
    new_image_list = []
    for im in image_list:
        newim = create_empty_image_like(im)
        newim.data = numpy.sqrt(numpy.abs(im.data))
        new_image_list.append(newim)
    return new_image_list

if __name__ == '__main__':
    
    logging.basicConfig(filename='simple-mpi.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    
    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    facets = 8
    assert facets * facets % size == 0
    
    # Create test image
    frequency = numpy.array([1e8])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    if rank == 0:
        model = create_test_image(frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                              polarisation_frame=PolarisationFrame('stokesI'))
    #f=show_image(model, title='Model image', cm='Greys', vmax=1.0, vmin=-0.1)
    #print(qa_image(model, context='Model image'))
    #plt.show()

    # Rank 0 scatters the test image
    if rank == 0:
        subimages = image_scatter_facets(model, facets=facets)
        subimages = numpy.array_split(subimages, size)
    else:
        subimages = list()
        
    sublist = comm.scatter(subimages, root=0)
    
    root_images = imagerooter(sublist)
    
    roots = comm.gather(root_images, root=0)
    
    if rank == 0:
        results = sum(roots, [])
        root_model = create_empty_image_like(model)
        result = image_gather_facets(results, root_model, facets=facets)
        numpy.testing.assert_array_almost_equal_nulp(result.data ** 2, numpy.abs(model.data), 7)
        print(qa_image(result))
