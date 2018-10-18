# Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# ARL Wrapping Interface
# In this file we declare the functions that need to be exposed to C (wrapped) --with the callback modifier
# and the needed structs -- with cdef

import numpy
import collections
import sys

from astropy.coordinates import SkyCoord
from astropy import units as u

from processing_components.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility, qa_gaintable
from processing_components.visibility.base import create_visibility, copy_visibility, create_visibility_from_rows
from processing_components.visibility.operations import qa_visibility
from data_models.memory_data_models import ReceptorFrame
from processing_components.image.deconvolution import deconvolve_cube, restore_cube
from processing_components.imaging.base import create_image_from_visibility, predict_2d, invert_2d, normalize_sumwt
from processing_components.imaging.wstack_single import invert_wstack_single, predict_wstack_single 
from processing_components.imaging.base import advise_wide_field
from processing_components.simulation.testing_support import create_named_configuration, create_test_image, create_low_test_image_from_gleam, simulate_gaintable
from data_models.polarisation import PolarisationFrame
from processing_components.visibility.base import create_blockvisibility
from workflows.serial.imaging.imaging_serial import invert_serial, predict_serial
from processing_components.image.operations import qa_image
from processing_components.visibility.coalesce import convert_visibility_to_blockvisibility, convert_blockvisibility_to_visibility
from processing_components.calibration.calibration import solve_gaintable
from workflows.serial.pipelines.pipeline_serial import ical_serial
from workflows.shared.imaging.imaging_shared import imaging_context
from data_models.data_model_helpers import export_image_to_hdf5

from ffiwrappers.src.arlwrap_support import *

import logging
import os

results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

arl_error = 0

def handle_error(*args):
    global arl_error
    if(args[0] != ""):
      arl_error = -1
      print(args[0],"\n",args[1],"\n",args[2])
 
ff.cdef("""
typedef struct {
  size_t nvis;
  int npol;
  void *data;
  char *phasecentre;
} ARLVis;
""")


ff.cdef("""
typedef struct {
  size_t nrows;
  void *data;
} ARLGt;
""")


ff.cdef("""
typedef struct {
  char *confname;
  double pc_ra;
  double pc_dec;
  double *times;
  int ntimes;
  double *freqs;
  int nfreqs;
  double *channel_bandwidth;
  int nchanwidth;
  int nbases;
  int nant;
  int npol;
  int nrec;
  double rmax;
  char *polframe;
} ARLConf;
""")



#@ff.callback("void (*)(const ARLVis *, ARLVis *, bool)")
#def arl_copy_visibility_ffi(visin, visout, zero):
#    """
#    Wrap of arl.visibility.base.copy_visibility
#    """
#    # Extra comments becasue this is an example.
#    #
#    # Convert the input visibilities into the ARL structure
#    nvisin=cARLVis(visin)
#
#    # Call the ARL function 
#    tvis=copy_visibility(nvisin, zero=zero)
#
#    # Copy the result into the output buffer
#    visout.npol=visin.npol
#    visout.nvis=visin.nvis
#    nvisout=cARLVis(visout)
#    numpy.copyto(nvisout, tvis)
#
#
#arl_copy_visibility=collections.namedtuple("FFIX", "address")    
#arl_copy_visibility.address=int(ff.cast("size_t", arl_copy_visibility_ffi))    

@ff.callback("int (*)()")
def arl_handle_error_ffi():

    global arl_error
    return arl_error

arl_handle_error=collections.namedtuple("FFIX", "address")    
arl_handle_error.address=int(ff.cast("size_t", arl_handle_error_ffi))    


@ff.callback("void (*)(ARLConf *, const ARLVis *, int, int *)")
def arl_create_rows_ffi(lowconfig, vis_in, vis_slices, c_rows):

# Create configuration object
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-create input blockvisibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_visin =  cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Create visibility object
    
    svis = convert_blockvisibility_to_visibility(py_visin)
     
    context='wstack'
    c = imaging_context(context)
    vis_iter = c['vis_iterator']   
    rows_tmp = numpy.empty(shape=(0), dtype =numpy.int32 )   
    print("vis_slices : ", vis_slices)
# Calculate rows for each vis_slice
    for rows in vis_iter(svis, vis_slices=vis_slices):
        rows_len = len(rows)        
        if numpy.sum(rows):
             print(numpy.sum(rows))
# Convert BOOL to INT and append rows to rows_tmp ndarray
             rows_tmp = numpy.append(rows_tmp, (rows*1).astype(numpy.int32))

    print("rows_tmp array length :", len(rows_tmp), rows_tmp.shape, type(rows_tmp))

#    for i in range(vis_slices):
#        print(i, rows_tmp[i*rows_len : i*rows_len + 10], numpy.sum(rows_tmp[i*rows_len : (i+1)*rows_len -1]))
    

# Copy rows_tmp to c_rows and return to C program
    rows_buf = numpy.frombuffer(ff.buffer(c_rows,
                                 vis_slices*rows_len*4),
                                 dtype=numpy.int32,
                                 count=vis_slices*rows_len)
    numpy.copyto(rows_buf, rows_tmp)
    #data = rows_tmp.__array_interface__['data'][0]
    #c_rows = ff.cast ( "int *" , data )
    #c_rows = ff.cast ( "int *" , rows_tmp.ctypes.data )

arl_create_rows=collections.namedtuple("FFIX", "address")    
arl_create_rows.address=int(ff.cast("size_t", arl_create_rows_ffi))    

@ff.callback("void (*)(ARLConf *, const ARLVis *, long long int *, ARLVis *, ARLVis *, long long int *, ARLVis *, int *)")
def arl_create_vis_from_rows_vis_ffi(lowconfig, vis_in, cindex_in, blockvis_in, vis_out, cindex_out, blockvis_out, c_rows):
# Create configuration object
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

# Re-create visibility object (+ cindex + vis.blockvis)
    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
    py_cindex = numpy.frombuffer(ff.buffer(cindex_in, 8*cindex_size), dtype='int', count=cindex_size)

    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    py_visin.cindex = py_cindex
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)


    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(vis_in.phasecentre) # Take the phasecentre from visibility part ( a mess!)
    py_blockvisin.configuration = lowcore
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    py_visin.blockvis = py_blockvisin

# Create rows ndarray for this vis_slice
    rows_int = numpy.frombuffer(ff.buffer(c_rows, py_visin.nvis*4),dtype=numpy.int32,count=py_visin.nvis)
    rows = rows_int.astype(bool)

# Create visslice from py_visin and rows
    visslice = create_visibility_from_rows(py_visin, rows)

# Link Visout object, visout.cindex and visout.blockvis
    py_visout = cARLVis_slice(vis_out, visslice.nvis)
#    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
#    py_cindex = numpy.frombuffer(ff.buffer(cindex_out, 8*cindex_size), dtype='int', count=cindex_size)
    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)

    numpy.copyto(py_visout, visslice.data)
    vis_out.nvis = visslice.nvis
    store_phasecentre(vis_out.phasecentre, py_visin.phasecentre)
# Copy vis.blockvis.data to C blockvisibility blockvis_out.data
    numpy.copyto(py_blockvis_out, visslice.blockvis.data)

# Copy vis.cindex to cindex_out
#    numpy.copyto(py_cindex, visslice.cindex)

arl_create_vis_from_rows_vis=collections.namedtuple("FFIX", "address")    
arl_create_vis_from_rows_vis.address=int(ff.cast("size_t", arl_create_vis_from_rows_vis_ffi))    


@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLVis *, long long int *, ARLVis *, int *)")
def arl_create_vis_from_rows_blockvis_ffi(lowconfig, vis_in, vis_out, cindex_out, blockvis_out, c_rows):

# Create configuration object
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-create input blockvisibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_visin =  cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Create visibility object
    svis = convert_blockvisibility_to_visibility(py_visin)
#    print("arl_create_vis_from_rows_blockvis_ffi nvis: ", svis.nvis)
#    print(type(svis.cindex), type(svis.blockvis))
   

# Create rows ndarray for this vis_slice
    rows_int = numpy.frombuffer(ff.buffer(c_rows, svis.nvis*4),dtype=numpy.int32,count=svis.nvis)
#    print(rows_int[0 : 10], numpy.sum(rows_int))

# Convert rows_int into boolean
    rows = rows_int.astype(bool)
#    print(rows[0 : 10])

#    print(len(rows), len(svis.cindex), len(svis.data), len(svis.blockvis.data))
#    print("arl_create_vis_from_rows_blockvis_ffi rows: ", rows[0:10])

# Create visslice from svis and rows
    visslice = create_visibility_from_rows(svis, rows)

# Copy vis data, blockvis data and cindex to the c arrays
#    print(len(visslice.data), visslice.size(), len(visslice.blockvis.data), visslice.blockvis.size(), sum(rows))
#    print(visslice)
#    print(visslice.blockvis)

# Link Visout object, visout.cindex and visout.blockvis
    py_visout = cARLVis_slice(vis_out, visslice.nvis)
#    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
#    py_cindex = numpy.frombuffer(ff.buffer(cindex_out, 8*cindex_size), dtype='int', count=cindex_size)
    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)

    numpy.copyto(py_visout, visslice.data)
    vis_out.nvis = visslice.nvis
    store_phasecentre(vis_out.phasecentre, py_visin.phasecentre)

# Copy vis.blockvis.data to C blockvisibility blockvis_out.data
    numpy.copyto(py_blockvis_out, visslice.blockvis.data)

#    print(qa_visibility(visslice, "arl_create_vis_from_rows_blockvis visslice"))

# Copy vis.cindex to cindex_out
#    numpy.copyto(py_cindex, visslice.cindex)

arl_create_vis_from_rows_blockvis=collections.namedtuple("FFIX", "address")    
arl_create_vis_from_rows_blockvis.address=int(ff.cast("size_t", arl_create_vis_from_rows_blockvis_ffi))    

#@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLVis *, int *)")
#def arl_create_vis_from_rows_vis_ffi(lowconfig, vis_in, vis_out, c_rows):

#arl_create_vis_from_rows_vis=collections.namedtuple("FFIX", "address")    
#arl_create_vis_from_rows_vis.address=int(ff.cast("size_t", arl_create_vis_from_rows_vis_ffi))    


@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLVis *, int)")
def arl_copy_visibility_ffi(lowconfig, vis_in, vis_out, zero_in):
# Convert the input blockvisibilities into the ARL structure

    if zero_in == 0:
         zero = True
    else:
         zero = False

# Create configuration object
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-create input blockvisibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_visin = cARLVis(vis_in)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

    # Call the ARL function 
    py_visout=copy_visibility(py_visin, zero=zero)

    # Copy the result into the output buffer
    vis_out.npol=vis_in.npol
    vis_out.nvis=vis_in.nvis

    py_vis_out = cARLVis(vis_out)
    numpy.copyto(py_vis_out, py_visout.data)
    store_phasecentre(vis_out.phasecentre, py_visin.phasecentre)

arl_copy_visibility=collections.namedtuple("FFIX", "address")    
arl_copy_visibility.address=int(ff.cast("size_t", arl_copy_visibility_ffi))    


@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLVis *, int)")
def arl_copy_blockvisibility_ffi(lowconfig, blockvis_in, blockvis_out, zero_in):
# Convert the input blockvisibilities into the ARL structure

    if zero_in == 0:
         zero = True
    else:
         zero = False

# Create configuration object
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-create input blockvisibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    # Call the ARL function 
    py_blockvisout=copy_visibility(py_blockvisin, zero=zero)

    # Copy the result into the output buffer
    blockvis_out.npol=blockvis_in.npol
    blockvis_out.nvis=blockvis_in.nvis

    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)
    numpy.copyto(py_blockvis_out, py_blockvisout.data)
    store_phasecentre(blockvis_out.phasecentre, py_blockvisin.phasecentre)


arl_copy_blockvisibility=collections.namedtuple("FFIX", "address")    
arl_copy_blockvisibility.address=int(ff.cast("size_t", arl_copy_blockvisibility_ffi))    

@ff.callback("void (*)(ARLConf *, ARLVis *)")
def arl_set_visibility_data_to_zero_ffi(lowconfig, vis_in):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

    py_visin.data['vis'][...] = 0.0

arl_set_visibility_data_to_zero=collections.namedtuple("FFIX", "address")
arl_set_visibility_data_to_zero.address=int(ff.cast("size_t", arl_set_visibility_data_to_zero_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, const ARLVis *, ARLVis *, int)")
def arl_manipulate_visibility_data_ffi(lowconfig, vis1_in, vis2_in, vis_out, operation):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_vis1in = cARLVis(vis1_in)
    py_vis1in = helper_create_visibility_object(c_vis1in)
    py_vis1in.phasecentre = load_phasecentre(vis1_in.phasecentre)
    py_vis1in.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_vis1in.polarisation_frame = PolarisationFrame(polframe)

    c_vis2in = cARLVis(vis2_in)
    py_vis2in = helper_create_visibility_object(c_vis2in)
    py_vis2in.phasecentre = load_phasecentre(vis2_in.phasecentre)
    py_vis2in.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_vis2in.polarisation_frame = PolarisationFrame(polframe)

    c_visout = cARLVis(vis_out)
    py_visout = helper_create_visibility_object(c_visout)
    py_visout.phasecentre = load_phasecentre(vis_out.phasecentre)
    py_visout.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visout.polarisation_frame = PolarisationFrame(polframe)
    print("arl_manipulate_visibility_data opcode: ", operation)
    if operation == 0:	# Add
         print("arl_manipulate_visibility_data: adding")
         py_visout.data['vis'] = py_vis1in.data['vis'] + py_vis2in.data['vis']
    elif operation == 1:	# Subtract
         print("arl_manipulate_visibility_data: subtracting")
         py_visout.data['vis'] = py_vis1in.data['vis'] - py_vis2in.data['vis']
    elif operation == 2:	# Multiply
         print("arl_manipulate_visibility_data: multiplying")
         py_visout.data['vis'] = py_vis1in.data['vis'] * py_vis2in.data['vis']
    elif operation == 3:	# Divide
         print("arl_manipulate_visibility_data: dividing")
         py_visout.data['vis'] = py_vis1in.data['vis'] / py_vis2in.data['vis']
    else:
         py_visout.data['vis'][...] = 0.0   

    print("arl_manipulate_visibility_data np.sum(vis.data): ", numpy.sum(py_visout.data['vis']), numpy.sum(py_vis1in.data['vis']), numpy.sum(py_vis2in.data['vis']))


arl_manipulate_visibility_data=collections.namedtuple("FFIX", "address")
arl_manipulate_visibility_data.address=int(ff.cast("size_t", arl_manipulate_visibility_data_ffi))

@ff.callback("void (*)(ARLConf *, ARLVis *, const ARLVis *)")
def arl_add_to_visibility_data_ffi(lowconfig, vis_inout, vis_in):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

    c_visout = cARLVis(vis_inout)
    py_visout = helper_create_visibility_object(c_visout)
    py_visout.phasecentre = load_phasecentre(vis_out.phasecentre)
    py_visout.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visout.polarisation_frame = PolarisationFrame(polframe)

    py_visout.data['vis'] += py_visin.data['vis']

arl_add_to_visibility_data=collections.namedtuple("FFIX", "address")
arl_add_to_visibility_data.address=int(ff.cast("size_t", arl_add_to_visibility_data_ffi))


@ff.callback("void (*)(ARLConf *, ARLVis *, const ARLVis *, int *)")
def arl_add_to_visibility_data_slice_ffi(lowconfig, vis_inout, vis_in, c_rows):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

    c_visout = cARLVis(vis_inout)
    py_visout = helper_create_visibility_object(c_visout)
    py_visout.phasecentre = load_phasecentre(vis_inout.phasecentre)
    py_visout.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visout.polarisation_frame = PolarisationFrame(polframe)

# Create rows ndarray for this vis_slice
    rows_int = numpy.frombuffer(ff.buffer(c_rows, py_visout.nvis*4),dtype=numpy.int32,count=py_visout.nvis)
    rows = rows_int.astype(bool)

    py_visout.data['vis'][rows] += py_visin.data['vis']
    print(qa_visibility(py_visout, "arl_add_to_visibility_data_slice"))

arl_add_to_visibility_data_slice=collections.namedtuple("FFIX", "address")
arl_add_to_visibility_data_slice.address=int(ff.cast("size_t", arl_add_to_visibility_data_slice_ffi))

ff.cdef("""
typedef struct {
    size_t size;
    int data_shape[4];
    void *data;
    char *wcs;
    char *polarisation_frame;
} Image;
""")


@ff.callback("void (*)(Image*, Image*)")
def arl_add_to_model_ffi(model, res):

    c_model = cImage(model)
    c_res = cImage(res)

    c_model.data += c_res.data

arl_add_to_model=collections.namedtuple("FFIX", "address")
arl_add_to_model.address=int(ff.cast("size_t", arl_add_to_model_ffi))


@ff.callback("void (*)(ARLConf *, ARLVis *)")
def arl_create_visibility_ffi(lowconfig, c_res_vis):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
# Temp fix for ffi_demo
    if lowconfig.rmax < 1.0e-5 :
       lowcore = create_named_configuration(lowcore_name)
    else:
       lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    print(lowcore_name)
    print("Times: ", times)
    print("Freqs: ", frequency)
    print("BW : ", channel_bandwidth)
    print("PCentre: ", lowconfig.pc_ra, lowconfig.pc_dec)

    phasecentre = SkyCoord(ra=lowconfig.pc_ra * u.deg, dec=lowconfig.pc_dec*u.deg, frame='icrs', 
            equinox='J2000')
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')

    vt = create_visibility(lowcore, times, frequency,
            channel_bandwidth=channel_bandwidth, weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame(polframe))

    py_res_vis = cARLVis(c_res_vis)

    numpy.copyto(py_res_vis, vt.data)

    store_phasecentre(c_res_vis.phasecentre, phasecentre)

arl_create_visibility=collections.namedtuple("FFIX", "address")
arl_create_visibility.address=int(ff.cast("size_t", arl_create_visibility_ffi))

@ff.callback("void (*)(ARLConf *, ARLVis *)",onerror=handle_error)
def arl_create_blockvisibility_ffi(lowconfig, c_res_vis):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    print(lowconfig.rmax)
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    print(lowcore_name)
    print("Times: ", times)
    print("Freqs: ", frequency)
    print("BW : ", channel_bandwidth)
    print("PCentre: ", lowconfig.pc_ra, lowconfig.pc_dec)

    phasecentre = SkyCoord(ra=lowconfig.pc_ra * u.deg, dec=lowconfig.pc_dec*u.deg, frame='icrs',
            equinox='J2000')

    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    print("Polarisation frame: ", polframe)
    vt = create_blockvisibility(lowcore, times, frequency=frequency,
            channel_bandwidth=channel_bandwidth, weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame(polframe))

    py_res_vis = cARLBlockVis(c_res_vis, lowconfig.nant, lowconfig.nfreqs)

    numpy.copyto(py_res_vis, vt.data)

    store_phasecentre(c_res_vis.phasecentre, phasecentre)

    receptor_frame = ReceptorFrame(vt.polarisation_frame.type)
    lowconfig.nrec = receptor_frame.nrec

arl_create_blockvisibility=collections.namedtuple("FFIX", "address")
arl_create_blockvisibility.address=int(ff.cast("size_t", arl_create_blockvisibility_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, const ARLVis *, long long int *, ARLVis *)")
def arl_convert_visibility_to_blockvisibility_ffi(lowconfig, vis_in, blockvis_in, cindex_in, blockvis_out):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
    py_cindex = numpy.frombuffer(ff.buffer(cindex_in, 8*cindex_size), dtype='int', count=cindex_size)

    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    py_visin.cindex = py_cindex
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)


    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    py_visin.blockvis = py_blockvisin

    py_blockvisout = convert_visibility_to_blockvisibility(py_visin)
#    print("convert_visibility_to_blockvisibility np.sum(block_vis.data): ", numpy.sum(py_blockvisout.data['vis']))
    print(qa_visibility(py_blockvisout, "arl_convert_visibility_to_blockvisibility"))

    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)
    numpy.copyto(py_blockvis_out, py_blockvisout.data)
    store_phasecentre(blockvis_out.phasecentre, py_blockvisin.phasecentre)

arl_convert_visibility_to_blockvisibility=collections.namedtuple("FFIX", "address")
arl_convert_visibility_to_blockvisibility.address=int(ff.cast("size_t", arl_convert_visibility_to_blockvisibility_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLVis *, long long int *, ARLVis *)")
def arl_convert_blockvisibility_to_visibility_ffi(lowconfig, blockvis_in, vis_out, cindex_out, blockvis_out):
# Create configuration object
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Link cindex memory objects
    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
    py_cindex = numpy.frombuffer(ff.buffer(cindex_out, 8*cindex_size), dtype='int', count=cindex_size)

# Re-create input blockvisibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

# Call arl.coalesce::convert_blockvisibility_to_visibility()
    vis = convert_blockvisibility_to_visibility(py_blockvisin)

# Copy vis.data to C visibility vis_out.data
    py_vis = cARLVis(vis_out)
    numpy.copyto(py_vis, vis.data)
    store_phasecentre(vis_out.phasecentre, py_blockvisin.phasecentre)

# Copy vis.blockvis.data to C blockvisibility blockvis_out.data
    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)
    numpy.copyto(py_blockvis_out, vis.blockvis.data)

# Copy vis.cindex to cindex_out
    numpy.copyto(py_cindex, vis.cindex)

    print("convert_blockvisibility_to_visibility np.sum(vis.data): ", numpy.sum(vis.data['vis']))



arl_convert_blockvisibility_to_visibility=collections.namedtuple("FFIX", "address")
arl_convert_blockvisibility_to_visibility.address=int(ff.cast("size_t", arl_convert_blockvisibility_to_visibility_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLGt *)")
def arl_create_gaintable_from_blockvisibility_ffi(lowconfig, blockvis_in, gt_out):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    py_gt = create_gaintable_from_blockvisibility(py_blockvisin)
#    print("create_gaintable_from_blockvisibility np.sum(gt.data): ", numpy.sum(py_gt.data['gain']))
#    print(py_gt.data['gain'].shape, py_gt.data['weight'].shape, py_gt.data['residual'].shape, py_gt.data['time'].shape)
#    print(py_gt.data.size, py_gt.data.itemsize)
#    print(py_gt.frequency.size)
#    print("create_gaintable_from_blockvisibility: ", py_gt.receptor_frame.nrec)

#    receptor_frame = ReceptorFrame(py_blockvisin.polarisation_frame.type)
#    pframe1 = PolarisationFrame(polframe)
#    recframe1 = ReceptorFrame(pframe1.type) 
#    print(receptor_frame.nrec, recframe1.nrec, lowcore.receptor_frame.nrec)
    c_gt_out = cARLGt(gt_out, lowconfig.nant, lowconfig.nfreqs, lowconfig.nrec)
    numpy.copyto(c_gt_out, py_gt.data)


arl_create_gaintable_from_blockvisibility=collections.namedtuple("FFIX", "address")
arl_create_gaintable_from_blockvisibility.address=int(ff.cast("size_t", arl_create_gaintable_from_blockvisibility_ffi))


@ff.callback("void (*)(ARLConf *, ARLGt *)")
def arl_simulate_gaintable_ffi(lowconfig, gt):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)

    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    polarisation_frame = PolarisationFrame(polframe)
    receptor_frame = ReceptorFrame(polarisation_frame.type)
#    print(lowconfig.polframe, lowconfig.nrec, receptor_frame.nrec)
    c_gt = cARLGt(gt, lowconfig.nant, lowconfig.nfreqs, lowconfig.nrec)
    py_gt = helper_create_gaintable_object(c_gt, frequency, receptor_frame)
    py_gt.receptor_frame = receptor_frame
#    print()
#    print(py_gt.__dict__)
#    print("simulate_gaintable 1 nrec: ", py_gt.receptor_frame.nrec)
#    print(py_gt.data['gain'].shape, py_gt.data['weight'].shape, py_gt.data['residual'].shape, py_gt.data['time'].shape)

#    py_gt = simulate_gaintable(py_gt, phase_error = 1.0)
    py_gt = simulate_gaintable(py_gt, phase_error = 0.0)

#    print("simulate_gaintable np.sum(gt.data): ", numpy.sum(py_gt.data['gain']))


#    print("simulate_gaintable 2 nrec: ", py_gt.receptor_frame.nrec)
#    print(py_gt.data['gain'].shape, py_gt.data['weight'].shape, py_gt.data['residual'].shape, py_gt.data['time'].shape)

    numpy.copyto(c_gt, py_gt.data)    

arl_simulate_gaintable=collections.namedtuple("FFIX", "address")
arl_simulate_gaintable.address=int(ff.cast("size_t", arl_simulate_gaintable_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLGt *, ARLVis *, int )")
def arl_apply_gaintable_ffi(lowconfig, blockvis_in, gt, blockvis_out, inverse_in):
    
    if inverse_in == 0:
         inverse = True
    else:
         inverse = False

    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

# Re-creating the input BlockVisibility object
    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating GainTable object
    receptor_frame = ReceptorFrame(py_blockvisin.polarisation_frame.type)
    c_gt = cARLGt(gt, lowconfig.nant, lowconfig.nfreqs, lowconfig.nrec)
    py_gt = helper_create_gaintable_object(c_gt, frequency, receptor_frame)
    py_gt.receptor_frame = receptor_frame

# Calling apply_gaintable() function
    py_blockvisout = apply_gaintable(py_blockvisin, py_gt, inverse=inverse)
#    print("apply_gaintable np.sum(blockvis.data): ", numpy.sum(py_blockvisout.data['vis']))
# Copy resulting data from py_blockvisout into c_blockvisout
    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)
    numpy.copyto(py_blockvis_out, py_blockvisout.data)
    store_phasecentre(blockvis_out.phasecentre, py_blockvisin.phasecentre)

arl_apply_gaintable=collections.namedtuple("FFIX", "address")
arl_apply_gaintable.address=int(ff.cast("size_t", arl_apply_gaintable_ffi))

@ff.callback("void (*)(ARLConf *, ARLVis *, ARLGt *, int )")
def arl_apply_gaintable_ical_ffi(lowconfig, blockvis_in, gt, inverse_in):
    
    if inverse_in == 0:
         inverse = True
    else:
         inverse = False

    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

# Re-creating the input BlockVisibility object
    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating GainTable object
    receptor_frame = ReceptorFrame(py_blockvisin.polarisation_frame.type)
    c_gt = cARLGt(gt, lowconfig.nant, lowconfig.nfreqs, lowconfig.nrec)
    py_gt = helper_create_gaintable_object(c_gt, frequency, receptor_frame)
    py_gt.receptor_frame = receptor_frame

# Calling apply_gaintable() function
    py_blockvisout = apply_gaintable(py_blockvisin, py_gt, inverse=inverse)
#    print("apply_gaintable np.sum(blockvis.data): ", numpy.sum(py_blockvisout.data['vis']))
# Copy resulting data from py_blockvisout back to c_blockvisin
    numpy.copyto(c_blockvisin, py_blockvisout.data)

arl_apply_gaintable_ical=collections.namedtuple("FFIX", "address")
arl_apply_gaintable_ical.address=int(ff.cast("size_t", arl_apply_gaintable_ical_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, const ARLVis *, ARLGt *, int )")
def arl_solve_gaintable_ical_ffi(lowconfig, blockvis_in, blockvis_pred, gt, vis_slices):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

# Re-creating the input BlockVisibility object
    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating the input BlockVisibility_pred object
    c_blockvispred = cARLBlockVis(blockvis_pred, lowconfig.nant, lowconfig.nfreqs)
    py_blockvispred = helper_create_blockvisibility_object(c_blockvispred, frequency, channel_bandwidth, lowcore)
    py_blockvispred.phasecentre = load_phasecentre(blockvis_pred.phasecentre)
    py_blockvispred.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvispred.polarisation_frame = PolarisationFrame(polframe)

# Re-creating GainTable object
    receptor_frame = ReceptorFrame(py_blockvisin.polarisation_frame.type)
    c_gt = cARLGt(gt, lowconfig.nant, lowconfig.nfreqs, lowconfig.nrec)
    py_gt = helper_create_gaintable_object(c_gt, frequency, receptor_frame)
    py_gt.receptor_frame = receptor_frame

# Calling apply_gaintable() function
    gt_out = solve_gaintable(py_blockvisin, py_blockvispred, 
	vis_slices=vis_slices, timeslice='auto',
        algorithm='hogbom', niter=1000, fractional_threshold=0.1, threshold=0.1,
        nmajor=5, gain=0.1, first_selfcal=1,
        global_solution=False)
    log.info(qa_gaintable(gt_out, context='Gaintable for selfcal cycle'))

    numpy.copyto(c_gt, gt_out.data)

#    print("apply_gaintable np.sum(blockvis.data): ", numpy.sum(py_blockvisout.data['vis']))

arl_solve_gaintable_ical=collections.namedtuple("FFIX", "address")
arl_solve_gaintable_ical.address=int(ff.cast("size_t", arl_solve_gaintable_ical_ffi))


ff.cdef("""
typedef struct {
	int vis_slices;
	int npixel;
	double cellsize;
	double guard_band_image;
	double delA;
	int wprojection_planes;
} ARLadvice ;
""")

@ff.callback("void (*)(ARLConf *, ARLVis *, ARLadvice *)")
def arl_advise_wide_field_ffi(lowconfig, vis_in, adv):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)
    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)

    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

    print("Index :", py_visin.data['index'])

    advice=advise_wide_field(py_visin, guard_band_image=adv.guard_band_image, delA=adv.delA,
                             wprojection_planes=adv.wprojection_planes)
    print(advice['vis_slices'], advice['npixels2'], advice['cellsize'])
    
    adv.cellsize = advice['cellsize']
    adv.vis_slices = advice['vis_slices']
    adv.npixel = advice['npixels2']
	
arl_advise_wide_field=collections.namedtuple("FFIX", "address")
arl_advise_wide_field.address=int(ff.cast("size_t", arl_advise_wide_field_ffi))


ff.cdef("""
typedef struct {int nant, nbases;} ant_t;
""")

# Get the number of baselines for the given configuration
# WARING!!! rmax is missing ! -ToDo
@ff.callback("void (*) (char*, ant_t *)")
def helper_get_nbases_ffi(config_name, nbases_in):
    tconfig_name = str(ff.string(config_name), 'utf-8')
    lowcore = create_named_configuration(tconfig_name)
    nbases_in.nant = len(lowcore.xyz)
    nbases_in.nbases = int(len(lowcore.xyz)*(len(lowcore.xyz)-1)/2)
    print(tconfig_name,nbases_in.nant, nbases_in.nbases )

helper_get_nbases=collections.namedtuple("FFIX", "address")    
helper_get_nbases.address=int(ff.cast("size_t", helper_get_nbases_ffi))  

# Get the number of baselines for the given configuration
# WARING!!! rmax is missing ! -ToDo
@ff.callback("void (*) (char*, double, ant_t *)")
def helper_get_nbases_rmax_ffi(config_name, rmax, nbases_in):
    tconfig_name = str(ff.string(config_name), 'utf-8')
    lowcore = create_named_configuration(tconfig_name, rmax=rmax)
    nbases_in.nant = len(lowcore.xyz)
    nbases_in.nbases = int(len(lowcore.xyz)*(len(lowcore.xyz)-1)/2)
    print(tconfig_name,nbases_in.nant, nbases_in.nbases )

helper_get_nbases_rmax=collections.namedtuple("FFIX", "address")    
helper_get_nbases_rmax.address=int(ff.cast("size_t", helper_get_nbases_rmax_ffi))  

@ff.callback("void (*)(ARLConf *, double, int, int *)")
def helper_get_image_shape_multifreq_ffi(lowconfig, cellsize, npixel, c_shape):
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    print("About to start create_low_test_image_from_gleam with flux_limit = 10. to get a shape of the image")
    res = create_low_test_image_from_gleam(npixel=npixel, frequency=frequency,
    channel_bandwidth=channel_bandwidth, cellsize=cellsize, flux_limit = 10.)
#	phasecentre=phasecentre, applybeam=True)
#    res = create_test_image(frequency=frequency, cellsize=cellsize, npixel = npixel)

    shape = list(res.data.shape)
    # TODO fix ugly
    numpy.copyto(numpy.frombuffer(ff.buffer(c_shape,4*4),dtype='i4',count=4), shape)

helper_get_image_shape_multifreq=collections.namedtuple("FFIX", "address")
helper_get_image_shape_multifreq.address=int(ff.cast("size_t", helper_get_image_shape_multifreq_ffi))

# TODO temporary until better solution found
@ff.callback("void (*)(const double *, double, int *)")
def helper_get_image_shape_ffi(freq, cellsize, c_shape):
    res = create_test_image(freq, cellsize)

    shape = list(res.data.shape)
    # TODO fix ugly
    numpy.copyto(numpy.frombuffer(ff.buffer(c_shape,4*4),dtype='i4',count=4), shape)

helper_get_image_shape=collections.namedtuple("FFIX", "address")
helper_get_image_shape.address=int(ff.cast("size_t", helper_get_image_shape_ffi))

# TODO properly implement this routine - shouldn't be within create_test_image
#@ff.callback("void (*)(const ARLVis *, Image *)")
#def helper_set_image_params_ffi(vis, image):
#    phasecentre = load_phasecentre(vis.phasecentre)
#
#    py_image = cImage(image)
#
#    py_image.wcs.wcs.crval[0] = phasecentre.ra.deg
#    py_image.wcs.wcs.crval[1] = phasecentre.dec.deg
#    py_image.wcs.wcs.crpix[0] = float(nx // 2)
#    py_image.wcs.wcs.crpix[1] = float(ny // 2)
#
#helper_set_image_params=collections.namedtuple("FFIX", "address")
#helper_set_image_params.address=int(ff.cast("size_t", helper_set_image_params_ffi))

@ff.callback("void (*)(const double *, double, char*, Image *)")
def arl_create_test_image_ffi(frequency, cellsize, c_phasecentre, out_img):
    py_outimg = cImage(out_img, new=True)

    res = create_test_image(frequency, cellsize)

    phasecentre = load_phasecentre(c_phasecentre)

    nchan, npol, ny, nx = res.data.shape

#    res.wcs.wcs.crval[0] = phasecentre.ra.deg
#    res.wcs.wcs.crval[1] = phasecentre.dec.deg
#    res.wcs.wcs.crpix[0] = float(nx // 2)
#    res.wcs.wcs.crpix[1] = float(ny // 2)

    store_image_in_c(py_outimg, res)

arl_create_test_image=collections.namedtuple("FFIX", "address")
arl_create_test_image.address=int(ff.cast("size_t", arl_create_test_image_ffi))

@ff.callback("void (*)(ARLConf *, double, int, char*, Image *)")
def arl_create_low_test_image_from_gleam_ffi(lowconfig, cellsize, npixel, c_phasecentre, out_img):
    py_outimg = cImage(out_img, new=True)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    phasecentre = load_phasecentre(c_phasecentre)
    print("About to start create_low_test_image_from_gleam")
    res = create_low_test_image_from_gleam(npixel=npixel, frequency=frequency,
    channel_bandwidth=channel_bandwidth, cellsize=cellsize, flux_limit = 1.0, phasecentre=phasecentre, applybeam=True)
    export_image_to_hdf5(res, '%s/gleam_model_res.hdf'%(results_dir))


    nchan, npol, ny, nx = res.data.shape

#    res.wcs.wcs.crval[0] = phasecentre.ra.deg
#    res.wcs.wcs.crval[1] = phasecentre.dec.deg
#    res.wcs.wcs.crpix[0] = float(nx // 2)
#    res.wcs.wcs.crpix[1] = float(ny // 2)

    export_image_to_hdf5(res, '%s/gleam_model_res1.hdf'%(results_dir))

    store_image_in_c(py_outimg, res)

arl_create_low_test_image_from_gleam=collections.namedtuple("FFIX", "address")
arl_create_low_test_image_from_gleam.address=int(ff.cast("size_t", arl_create_low_test_image_from_gleam_ffi))


@ff.callback("void (*)(const ARLVis *, const Image *, ARLVis *)")
def arl_predict_2d_ffi(vis_in, img, vis_out):
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    c_img = cImage(img)

    
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)

    res = predict_2d(py_visin, c_img)

    vis_out.nvis = vis_in.nvis
    vis_out.npol = vis_in.npol
    c_visout = cARLVis(vis_out)

    numpy.copyto(c_visout, res.data)
    store_phasecentre(vis_out.phasecentre, res.phasecentre)

    #arl_copy_visibility(py_visin, c_visout, False)

arl_predict_2d=collections.namedtuple("FFIX", "address")
arl_predict_2d.address=int(ff.cast("size_t", arl_predict_2d_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, const Image *, ARLVis *, ARLVis *, long long int *, int)",onerror=handle_error)
def arl_predict_function_ffi(lowconfig, vis_in, img, vis_out, blockvis_out, cindex_out, vis_slices):

    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
    py_cindex = numpy.frombuffer(ff.buffer(cindex_out, 8*cindex_size), dtype='int', count=cindex_size)

    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    c_img = cImage(img)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

#    print("--------------------> predict_serial Phasecentre : ", py_visin.phasecentre.ra.deg, py_visin.phasecentre.dec.deg)

    res = predict_serial(py_visin, c_img, vis_slices=vis_slices, context='wstack')
#    print("--------------------> predict_serial sizeof(py_visin.data), sizeof(res.data)", sys.getsizeof(py_visin.data[:]), sys.getsizeof(res.data[:]))
#    print("--------------------> predict_serial cindex",  type(res.cindex), type(res.cindex[0]), len(res.cindex))
#    print("--------------------> predict_serial sys.getsizeof(res.cindex)", sys.getsizeof(res.cindex))

#    print("--------------------> predict_serial np.sum(predicted_vis.data): ", numpy.sum(res.data['vis']))
#    print("--------------------> predict_serial predicted_vis.data: ", res.data)
#    print("--------------------> predict_serial py_visin.data): ", py_visin.data)

#    print("predict_serial np.sum(predicted_vis.data): ", numpy.sum(res.data['vis']))

    vis_out.npol = vis_in.npol
    c_visout = cARLVis(vis_out)
    numpy.copyto(c_visout, res.data)
    store_phasecentre(vis_out.phasecentre, res.phasecentre)
    numpy.copyto(py_cindex, res.cindex)

    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)
    numpy.copyto(py_blockvis_out, res.blockvis.data)
    store_phasecentre(blockvis_out.phasecentre, res.phasecentre)

arl_predict_function=collections.namedtuple("FFIX", "address")
arl_predict_function.address=int(ff.cast("size_t", arl_predict_function_ffi))

@ff.callback("void (*)(ARLConf *, ARLVis *, const Image *, int)",onerror=handle_error)
def arl_predict_function_blockvis_ffi(lowconfig, vis_in, img, vis_slices):

    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    c_img = cImage(img)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)

    log.info(qa_image(c_img, context='arl_predict_function_blockvis'))

#    export_image_to_fits(c_img, '%s/imaging-blockvis_model_in_predicted_function.fits'%(results_dir))
#    export_blockvisibility_to_hdf5(py_visin, '%s/py_visin.hdf'%(results_dir))
#    export_image_to_hdf5(c_img, '%s/gleam_model_c_img.hdf'%(results_dir))

    py_blockvis = predict_serial(py_visin, c_img, vis_slices=vis_slices, context='wstack')

#    export_blockvisibility_to_hdf5(py_blockvis, '%s/py_blockvis.hdf'%(results_dir))

    print(qa_visibility(py_blockvis, context='arl_predict_function_blockvis py_blockvis'))
#    print("arl_predict_function_blockvis :", py_visin, py_blockvis)

    numpy.copyto(c_visin, py_blockvis.data)
#    store_phasecentre(vis_out.phasecentre, res.phasecentre)

#    print("arl_predict_function_blockvis np.sum(py_blockvis.data): ", numpy.sum(py_blockvis.data['vis']))
#    print("arl_predict_function_blockvis nchan npol nants ", py_blockvis.nchan, py_blockvis.npol, py_blockvis.nants)
#    print("arl_predict_function_blockvis sum(uvw) ", numpy.sum(py_blockvis.uvw))
#    print("arl_predict_function_blockvis sum(vis) ", numpy.sum(py_blockvis.vis))
#    print("arl_predict_function_blockvis sum(weight) ", numpy.sum(py_blockvis.weight))
#    print("arl_predict_function_blockvis time", py_blockvis.time, numpy.sum(py_blockvis.time))
#    print("arl_predict_function_blockvis integration_time", py_blockvis.integration_time, numpy.sum(py_blockvis.integration_time))
#    print("arl_predict_function_blockvis nvis, size", py_blockvis.nvis, py_blockvis.size())

arl_predict_function_blockvis=collections.namedtuple("FFIX", "address")
arl_predict_function_blockvis.address=int(ff.cast("size_t", arl_predict_function_blockvis_ffi))

@ff.callback("void (*)(ARLConf *, ARLVis *, const Image *, ARLVis *, long long int *, int)")
def arl_predict_function_ical_ffi(lowconfig, vis_inout, img, blockvis_inout, cindex_inout, vis_slices):

    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
    py_cindex = numpy.frombuffer(ff.buffer(cindex_inout, 8*cindex_size), dtype='int', count=cindex_size)

    c_visinout = cARLVis(vis_inout)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    py_visinout = helper_create_visibility_object(c_visinout)
    py_visinout.configuration = lowcore

    py_visinout.phasecentre = load_phasecentre(vis_inout.phasecentre)
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visinout.polarisation_frame = PolarisationFrame(polframe)

    py_blockvis_inout = cARLBlockVis(blockvis_inout, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisinout = helper_create_blockvisibility_object(py_blockvis_inout, frequency, channel_bandwidth, lowcore)
    py_visinout.blockvis = py_blockvisinout
    py_visinout.cindex = py_cindex

    c_img = cImage(img)

    res = predict_serial(py_visinout, c_img, vis_slices=vis_slices, context='wstack', 
                    timeslice='auto', algorithm='hogbom', niter=1000, fractional_threshold=0.1,
                    threshold=0.1, nmajor=5, gain=0.1, first_selfcal=1, global_solution=False)
#    print("####################> arl_predict_function_ical: ", type(res))

    numpy.copyto(c_visinout, res.data)
    store_phasecentre(vis_inout.phasecentre, res.phasecentre)
    numpy.copyto(py_cindex, res.cindex)

    numpy.copyto(py_blockvis_inout, res.blockvis.data)
    store_phasecentre(blockvis_inout.phasecentre, res.phasecentre)
#    print("predict_function_ical np.sum(res.data): ", numpy.sum(res.data['vis']))
#    print("predict_function_ical np.sum(res.blockvis.data): ", numpy.sum(res.blockvis.data['vis']))

arl_predict_function_ical=collections.namedtuple("FFIX", "address")
arl_predict_function_ical.address=int(ff.cast("size_t", arl_predict_function_ical_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, const ARLVis *, const Image *, ARLVis *)")
def arl_predict_function_oneslice_ffi(lowconfig, vis_in, blockvis_in, img, vis_out):
# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-creating input Visibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating vis.BlockVisibility object
    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    py_visin.blockvis = py_blockvisin

# Re-creating image object
    # Re-creating images 
    py_img = cImage(img)

# Re-creating output Visibility object
    c_visout = cARLVis(vis_out)
#    py_visout = helper_create_visibility_object(c_visout)
#    py_visout.phasecentre = load_phasecentre(vis_out.phasecentre)
#    py_visout.configuration = lowcore
#    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
#    py_visout.polarisation_frame = PolarisationFrame(polframe)

# Get predicted visibilities
    result = predict_wstack_single(py_visin, py_img)
# Copy visibility data to vis_out
    numpy.copyto(c_visout, result.data)
    

arl_predict_function_oneslice=collections.namedtuple("FFIX", "address")
arl_predict_function_oneslice.address=int(ff.cast("size_t", arl_predict_function_oneslice_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, Image *, int, Image *)")
def arl_invert_function_ffi(lowconfig, vis_in, img, vis_slices, img_dirty):
# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-creating Visibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating images 
    py_img = cImage(img)
    py_img_dirty = cImage(img_dirty, new=True)

# Calling invert_finction()
#    export_blockvisibility_to_hdf5(py_visin, '%s/py_visin_invert_function.hdf'%(results_dir))
#    export_image_to_hdf5(py_img, '%s/model_invert_function.hdf'%(results_dir))
#    print("arl_invert_function vis_slices: ", vis_slices)

    dirty, sumwt = invert_serial(py_visin, py_img, vis_slices=vis_slices, dopsf=False, context='wstack')
    nchan, npol, ny, nx = dirty.data.shape

#    dirty.wcs.wcs.crval[0] = py_visin.phasecentre.ra.deg
#    dirty.wcs.wcs.crval[1] = py_visin.phasecentre.dec.deg
#    dirty.wcs.wcs.crpix[0] = float(nx // 2)
#    dirty.wcs.wcs.crpix[1] = float(ny // 2)

# Copy Python dirty image into C image
    store_image_in_c(py_img_dirty, dirty)

arl_invert_function=collections.namedtuple("FFIX", "address")
arl_invert_function.address=int(ff.cast("size_t", arl_invert_function_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, const ARLVis *, Image *, int, Image *, double *, int)")
def arl_invert_function_oneslice_ffi(lowconfig, vis_in, blockvis_in, img, vis_slices, img_dirty, c_sumwt, dopsf_in):
# Get dopsf
    if dopsf_in == 0:
        dopsf = True
    else:
        dopsf = False

# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-creating Visibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating BlockVisibility object
    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    py_visin.blockvis = py_blockvisin
#    print(qa_visibility(py_visin, "arl_invert_function_oneslice py_visin"))
#    print(py_visin)
# Re-creating images 
    py_img = cImage(img)
    py_img_dirty = cImage(img_dirty, new=True)
#    print(qa_image(py_img, "arl_invert_function_oneslice py_img"))

# Calling invert_finction()
#    export_blockvisibility_to_hdf5(py_visin, '%s/py_visin_invert_function.hdf'%(results_dir))
#    export_image_to_hdf5(py_img, '%s/model_invert_function.hdf'%(results_dir))
#    print("arl_invert_function vis_slices: ", vis_slices)

    dirty, sumwt = invert_wstack_single(py_visin, py_img, vis_slices=1, dopsf=dopsf, context='wstack', normalize=False)
    nchan, npol, ny, nx = dirty.data.shape
#    print("arl_invert_function_oneslice sumwt: ", sumwt, sumwt.shape)
# Copy sumwt to c_sumwt (total weight for the normalization)
    py_sumwt = numpy.frombuffer(ff.buffer(c_sumwt, nchan*npol*8),dtype=numpy.float64,count=nchan*npol)
    numpy.copyto(py_sumwt, sumwt)

#    dirty.wcs.wcs.crval[0] = py_visin.phasecentre.ra.deg
#    dirty.wcs.wcs.crval[1] = py_visin.phasecentre.dec.deg
#    dirty.wcs.wcs.crpix[0] = float(nx // 2)
#    dirty.wcs.wcs.crpix[1] = float(ny // 2)

# Copy Python dirty image into C image
    store_image_in_c(py_img_dirty, dirty)
#    print(qa_image(dirty, "arl_invert_function_oneslice dirty"))

arl_invert_function_oneslice=collections.namedtuple("FFIX", "address")
arl_invert_function_oneslice.address=int(ff.cast("size_t", arl_invert_function_oneslice_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, Image *, int, Image *)")
def arl_invert_function_blockvis_ffi(lowconfig, vis_in, img, vis_slices, img_dirty):
# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-creating Visibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating images 
    py_img = cImage(img)
    py_img_dirty = cImage(img_dirty, new=True)
# Calling invert_finction()
#    export_blockvisibility_to_hdf5(py_visin, '%s/py_visin_invert_function.hdf'%(results_dir))
#    export_image_to_hdf5(py_img, '%s/model_invert_function.hdf'%(results_dir))
#    print("arl_invert_function vis_slices: ", vis_slices)
    dirty, sumwt = invert_serial(py_visin, py_img, vis_slices=vis_slices, dopsf=False, context='wstack')
    nchan, npol, ny, nx = dirty.data.shape

#    dirty.wcs.wcs.crval[0] = py_visin.phasecentre.ra.deg
#    dirty.wcs.wcs.crval[1] = py_visin.phasecentre.dec.deg
#    dirty.wcs.wcs.crpix[0] = float(nx // 2)
#    dirty.wcs.wcs.crpix[1] = float(ny // 2)

# Copy Python dirty image into C image
    store_image_in_c(py_img_dirty, dirty)

arl_invert_function_blockvis=collections.namedtuple("FFIX", "address")
arl_invert_function_blockvis.address=int(ff.cast("size_t", arl_invert_function_blockvis_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, Image *, int, Image *)")
def arl_invert_function_ical_ffi(lowconfig, vis_in, img, vis_slices, img_dirty):
# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-creating Visibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating images 
    py_img = cImage(img)
    py_img_dirty = cImage(img_dirty, new=True)
# Calling invert_finction()
    dirty, sumwt = invert_serial(py_visin, py_img, vis_slices=vis_slices, context='wstack',
                    timeslice='auto', algorithm='hogbom', niter=1000, fractional_threshold=0.1,
                    threshold=0.1, nmajor=5, gain=0.1, first_selfcal=1,global_solution=False)

    nchan, npol, ny, nx = dirty.data.shape

#    dirty.wcs.wcs.crval[0] = py_visin.phasecentre.ra.deg
#    dirty.wcs.wcs.crval[1] = py_visin.phasecentre.dec.deg
#    dirty.wcs.wcs.crpix[0] = float(nx // 2)
#    dirty.wcs.wcs.crpix[1] = float(ny // 2)

# Copy Python dirty image into C image
    store_image_in_c(py_img_dirty, dirty)
    log.info("Maximum in residual image is %.6f" % (numpy.max(numpy.abs(dirty.data))))

arl_invert_function_ical=collections.namedtuple("FFIX", "address")
arl_invert_function_ical.address=int(ff.cast("size_t", arl_invert_function_ical_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, Image *, int, Image *)")
def arl_invert_function_psf_ffi(lowconfig, vis_in, img, vis_slices, img_psf):
# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

# Re-creating Visibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating images 
    py_img = cImage(img)
    py_img_psf = cImage(img_psf, new=True)
# Calling invert_finction()
    psf, sumwt = invert_serial(py_visin, py_img, vis_slices=vis_slices, dopsf=True, context='wstack',
                    timeslice='auto', algorithm='hogbom', niter=1000, fractional_threshold=0.1,
                    threshold=0.1, nmajor=5, gain=0.1, first_selfcal=1,global_solution=False)

    nchan, npol, ny, nx = psf.data.shape

#    psf.wcs.wcs.crval[0] = py_visin.phasecentre.ra.deg
#    psf.wcs.wcs.crval[1] = py_visin.phasecentre.dec.deg
#    psf.wcs.wcs.crpix[0] = float(nx // 2)
#    psf.wcs.wcs.crpix[1] = float(ny // 2)

# Copy Python dirty image into C image
    store_image_in_c(py_img_psf, psf)

arl_invert_function_psf=collections.namedtuple("FFIX", "address")
arl_invert_function_psf.address=int(ff.cast("size_t", arl_invert_function_psf_ffi))

@ff.callback("void (*)(Image *, double *)")
def arl_normalize_sumwt_ffi(dirty, sumwt):
    c_dirty = cImage(dirty)
    nchan, npol, ny, nx = c_dirty.data.shape
    print("arl_normalize_sumwt nchan, npol: ", nchan, npol)

# Copy sumwt to c_sumwt (total weight for the normalization)
    py_sumwt = numpy.frombuffer(ff.buffer(sumwt, nchan*npol*8),dtype=numpy.float64,count=nchan*npol)
    py_sumwt = py_sumwt.reshape((nchan,npol))
    print("arl_normalize_sumwt py_sumwt: ", py_sumwt, py_sumwt.shape)
    
    py_result = normalize_sumwt(c_dirty, py_sumwt)

    numpy.copyto(c_dirty.data, py_result.data)
#    store_image_in_c(c_dirty, py_result)

arl_normalize_sumwt=collections.namedtuple("FFIX", "address")    
arl_normalize_sumwt.address=int(ff.cast("size_t", arl_normalize_sumwt_ffi))    


@ff.callback("void (*)(ARLConf *, const ARLVis *, Image *, int, Image *, Image *, Image *)")
def arl_ical_ffi(lowconfig, blockvis_in, img_model, vis_slices, img_deconvolved, img_residual, img_restored):
# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)
# Re-creating BlockVisibility object
    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

# Re-creating images 
    py_model 		= cImage(img_model)
    py_img_deconvolved 	= cImage(img_deconvolved, new=True)
    py_img_residual 	= cImage(img_residual, new=True)
    py_img_restored 	= cImage(img_restored, new=True)

# Callinc ical_serial()
    deconvolved, residual, restored = ical_serial(block_vis=py_blockvisin, model=py_model, vis_slices=vis_slices,
                                            timeslice='auto',
                                                  algorithm='hogbom', niter=1000, fractional_threshold=0.1, threshold=0.1,
                                                  context='wstack', nmajor=5, gain=0.1, first_selfcal=1,
                                                  global_solution=False)

# Preparing deconvolved
    nchan, npol, ny, nx = deconvolved.data.shape
#    deconvolved.wcs.wcs.crval[0] = py_blockvisin.phasecentre.ra.deg
#    deconvolved.wcs.wcs.crval[1] = py_blockvisin.phasecentre.dec.deg
#    deconvolved.wcs.wcs.crpix[0] = float(nx // 2)
#    deconvolved.wcs.wcs.crpix[1] = float(ny // 2)
    store_image_in_c(py_img_deconvolved, deconvolved)

# Preparing residual
    nchan, npol, ny, nx = residual.data.shape
#    residual.wcs.wcs.crval[0] = py_blockvisin.phasecentre.ra.deg
#    residual.wcs.wcs.crval[1] = py_blockvisin.phasecentre.dec.deg
#    residual.wcs.wcs.crpix[0] = float(nx // 2)
#    residual.wcs.wcs.crpix[1] = float(ny // 2)
    store_image_in_c(py_img_residual, residual)

# Preparing restored
    nchan, npol, ny, nx = restored.data.shape
#    restored.wcs.wcs.crval[0] = py_blockvisin.phasecentre.ra.deg
#    restored.wcs.wcs.crval[1] = py_blockvisin.phasecentre.dec.deg
#    restored.wcs.wcs.crpix[0] = float(nx // 2)
#    restored.wcs.wcs.crpix[1] = float(ny // 2)
    store_image_in_c(py_img_restored, restored)

arl_ical=collections.namedtuple("FFIX", "address")
arl_ical.address=int(ff.cast("size_t", arl_ical_ffi))


@ff.callback("void (*)(const ARLVis *, const Image *, bool dopsf, Image *, double *)")
def arl_invert_2d_ffi(invis, in_image, dopsf, out_image, sumwt):
    py_visin = helper_create_visibility_object(cARLVis(invis))
    c_in_img = cImage(in_image)
    c_out_img = cImage(out_image, new=True)
    py_visin.phasecentre = load_phasecentre(invis.phasecentre)

    if dopsf:
        out, sumwt = invert_2d(py_visin, c_in_img, dopsf=True)
    else:
        out, sumwt = invert_2d(py_visin, c_in_img)


    store_image_in_c_2(c_out_img, out)

arl_invert_2d=collections.namedtuple("FFIX", "address")
arl_invert_2d.address=int(ff.cast("size_t", arl_invert_2d_ffi))

@ff.callback("void (*)(const ARLVis *, Image *)")
def arl_create_image_from_visibility_ffi(vis_in, img_in):
    c_vis = cARLVis(vis_in)
    c_img = cImage(img_in, new=True);

    # We need a proper Visibility object - not this, and not a cARLVis
    # This is temporary - just so we have some data to pass to
    # the create_... routine
    tvis = helper_create_visibility_object(c_vis)
    tvis.phasecentre = load_phasecentre(vis_in.phasecentre)

    # Default args for now
    image = create_image_from_visibility(tvis, cellsize=0.001, npixel=256)

    #numpy.copyto(c_img.data, image.data)

    # Pickle WCS and polframe, until better way is found to handle these data
    # structures
    #store_image_pickles(c_img, image)
    store_image_in_c(c_img, image)


arl_create_image_from_visibility=collections.namedtuple("FFIX", "address")    
arl_create_image_from_visibility.address=int(ff.cast("size_t",
    arl_create_image_from_visibility_ffi))    

@ff.callback("void (*)(ARLConf *, const ARLVis *, double, int, char*, Image *)")
def arl_create_image_from_blockvisibility_ffi(lowconfig, blockvis_in, cellsize, npixel, c_phasecentre, img_out):
# Creating configuration
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)
# Re-creating BlockVisibility object
    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
#    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
# Copying phasecentre and other metadata
    phasecentre = load_phasecentre(c_phasecentre)
    py_blockvisin.phasecentre = phasecentre
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    phasecentre1 = SkyCoord(ra=lowconfig.pc_ra * u.deg, dec=lowconfig.pc_dec*u.deg, frame='icrs',
            equinox='J2000')

# Re-creating Image object
    py_outimg = cImage(img_out, new=True);

# Construct a model from py_blockvisin
    res = create_image_from_visibility(py_blockvisin, npixel=npixel, frequency=[numpy.average(frequency)], nchan=1,
        channel_bandwidth=[numpy.sum(channel_bandwidth)], cellsize=cellsize, phasecentre=phasecentre1)

    #numpy.copyto(c_img.data, image.data)

    # Pickle WCS and polframe, until better way is found to handle these data
    # structures
    #store_image_pickles(c_img, image)
    nchan, npol, ny, nx = res.data.shape

#    res.wcs.wcs.crval[0] = phasecentre1.ra.deg
#    res.wcs.wcs.crval[1] = phasecentre1.dec.deg
#    res.wcs.wcs.crpix[0] = float(nx // 2)
#    res.wcs.wcs.crpix[1] = float(ny // 2)

    store_image_in_c(py_outimg, res)


arl_create_image_from_blockvisibility=collections.namedtuple("FFIX", "address")    
arl_create_image_from_blockvisibility.address=int(ff.cast("size_t",
    arl_create_image_from_blockvisibility_ffi))    

@ff.callback("void (*)(Image *, Image *, Image *, Image *)")
def arl_deconvolve_cube_ffi(dirty, psf, restored, residual):
    c_dirty = cImage(dirty)
    c_psf = cImage(psf)
    c_residual = cImage(residual, new=True)
    c_restored = cImage(restored, new=True)

    py_restored, py_residual = deconvolve_cube(c_dirty, c_psf,
            niter=1000,threshold=0.001, fracthresh=0.01, window_shape='quarter',
            gain=0.7, scales=[0,3,10,30])

    store_image_in_c(c_restored,py_restored)
    store_image_in_c(c_residual,py_residual)


arl_deconvolve_cube=collections.namedtuple("FFIX", "address")    
arl_deconvolve_cube.address=int(ff.cast("size_t", arl_deconvolve_cube_ffi))    

@ff.callback("void (*)(Image *, Image *, Image *, Image *)")
def arl_deconvolve_cube_ical_ffi(dirty, psf, restored, residual):
    c_dirty = cImage(dirty)
    c_psf = cImage(psf)
    c_residual = cImage(residual, new=True)
    c_restored = cImage(restored, new=True)

    py_restored, py_residual = deconvolve_cube(c_dirty, c_psf,
                   timeslice='auto', algorithm='hogbom', niter=1000, fractional_threshold=0.1,
                   threshold=0.1, nmajor=5, gain=0.1, first_selfcal=1, global_solution=False)

    store_image_in_c(c_restored,py_restored)
    store_image_in_c(c_residual,py_residual)


arl_deconvolve_cube_ical=collections.namedtuple("FFIX", "address")    
arl_deconvolve_cube_ical.address=int(ff.cast("size_t", arl_deconvolve_cube_ical_ffi))    

@ff.callback("void (*)(Image *, Image *, Image*, Image*)")
def arl_restore_cube_ffi(model, psf, residual, restored):
    # Cast C Image structs to Python objects
    c_model = cImage(model)
    c_psf = cImage(psf)
    if residual:
        c_residual = cImage(residual)
    else:
        c_residual = None
    c_restored = cImage(restored, new=True)

    # Calculate
    py_restored = restore_cube(c_model, c_psf, c_residual)

    # Copy Python result to C result struct
    store_image_in_c(c_restored,py_restored)

arl_restore_cube=collections.namedtuple("FFIX", "address")    
arl_restore_cube.address=int(ff.cast("size_t", arl_restore_cube_ffi))    

@ff.callback("void (*)(Image *, Image *, Image*, Image*)")
def arl_restore_cube_ical_ffi(model, psf, residual, restored):
    # Cast C Image structs to Python objects
    c_model = cImage(model)
    c_psf = cImage(psf)
    if residual:
        c_residual = cImage(residual)
    else:
        c_residual = None
    c_restored = cImage(restored, new=True)

    # Calculate
    py_restored = restore_cube(c_model, c_psf, c_residual,
                   timeslice='auto', algorithm='hogbom', niter=1000, fractional_threshold=0.1,
                   threshold=0.1, nmajor=5, gain=0.1, first_selfcal=1, global_solution=False)


    # Copy Python result to C result struct
    store_image_in_c(c_restored,py_restored)

arl_restore_cube_ical=collections.namedtuple("FFIX", "address")    
arl_restore_cube_ical.address=int(ff.cast("size_t", arl_restore_cube_ical_ffi))    


