# -*- coding: utf-8 -*-
"""Read OSKAR binary files from python."""

import struct
import collections
import numpy
import os

from astropy.coordinates import ICRS, EarthLocation
from astropy.wcs import WCS

from arl.data.data_models import *


class OskarBinary(object):

    """Class providing an interface to OSKAR binary data files.

    see:
        http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Binary-File-Format.pdf

    TODO:
        - Split data reading from indexing to be able to deal with very large
          files and make reading a sub-set of the data faster.
    """

    # noinspection PyRedeclaration
    class DataType:
        # noinspection PyRedeclaration
        Char, Int, Single, Double, _, Complex, Matrix, _ = range(8)

    # noinspection PyRedeclaration,PyRedeclaration,PyRedeclaration,PyRedeclaration
    class Group:
        # noinspection PyRedeclaration,PyRedeclaration,PyRedeclaration,PyRedeclaration
        _, Standard, _, Settings, RunInfo, _, _,\
            Sky, _, Spline, Element, VisHeader, VisBlock = range(13)

    class Standard:
        _, DateTime, Version, UserName, WorkingDir = range(5)

    class Settings(object):
        Path = 1
        File = 2

    class RunInfo(object):
        Log = 1

    def __init__(self, file_name):
        """Constructor."""
        if not os.path.exists(file_name):
            raise ValueError('Specified visibility file %s not found!' % file_name)
        self.file_name = file_name
        self.file_handle = open(file_name, 'rb')
        self.bin_ver = 0
        self.record = collections.OrderedDict()
        self.read()

    def __del__(self):
        """Destructor."""
        self.file_handle.close()

    def read_header(self):
        """Read header."""
        f = self.file_handle
        name = f.read(9)
        if name[0:8] != b'OSKARBIN':
            raise ValueError('Not a valid OSKAR binary file.')
        bin_ver = struct.unpack('B', f.read(1))[0]
        if not (bin_ver == 1 or bin_ver == 2):
            raise ValueError('The class can only read OSKAR binary '
                               'format version 1 or 2.')
        self.bin_ver = bin_ver

        # Version 1: header information.
        if bin_ver == 1:
            endian = struct.unpack('B', f.read(1))[0]
            svoid = struct.unpack('B', f.read(1))[0]
            sint = struct.unpack('B', f.read(1))[0]
            slong = struct.unpack('B', f.read(1))[0]
            sfloat = struct.unpack('B', f.read(1))[0]
            sdouble = struct.unpack('B', f.read(1))[0]
            patch = struct.unpack('B', f.read(1))[0]
            minor = struct.unpack('B', f.read(1))[0]
            major = struct.unpack('B', f.read(1))[0]
            other = struct.unpack('B', f.read(1))[0]
        # Version 2: read remaining reserved space.
        else:
            _ = f.read(64 - 10)

    @staticmethod
    def is_set(x, n):
        """Checks if a flag is set (value of bit n in byte x)."""
        return x & 2**n != 0

    def read_block_header(self, block_index):
        """."""
        f = self.file_handle

        element_size = struct.unpack('B', f.read(1))[0]
        chunk_flags = struct.unpack('B', f.read(1))[0]
        data_type = struct.unpack('B', f.read(1))[0]
        group = struct.unpack('B', f.read(1))[0]
        tag = struct.unpack('B', f.read(1))[0]
        index = struct.unpack('i', f.read(4))[0]
        block_size = struct.unpack('l', f.read(8))[0]

        if group not in self.record:
            self.record[group] = collections.OrderedDict()
        if tag not in self.record[group]:
            self.record[group][tag] = collections.OrderedDict()
        if index not in self.record[group][tag]:
            self.record[group][tag][index] = collections.OrderedDict()

        block = self.record[group][tag][index]
        block['group'] = group
        block['tag'] = tag
        block['index'] = index
        block['number'] = block_index
        block['element_size'] = element_size
        block['chunk_flags'] = chunk_flags
        block['flag_endian'] = self.is_set(chunk_flags, 5)
        block['flag_crc'] = self.is_set(chunk_flags, 6)
        block['flag_extended'] = self.is_set(chunk_flags, 7)
        block['data_type'] = data_type
        block['block_size'] = block_size

        return block

    def read_block_data(self, block):
        """."""
        f = self.file_handle

        # Data size of the block payload.
        data_size = block['block_size']
        if block['flag_crc']:
            data_size -= 4

        # Read the block payload.
        if self.is_set(block['data_type'], self.DataType.Char):
            name = 'char'
            n = data_size
            data = f.read(data_size)

        elif self.is_set(block['data_type'], self.DataType.Int):
            name = 'int'
            n = data_size // block['element_size']
            data = struct.unpack('i' * n, f.read(data_size))

        elif self.is_set(block['data_type'], self.DataType.Single):
            if self.is_set(block['data_type'], self.DataType.Matrix):
                if self.is_set(block['data_type'], self.DataType.Complex):
                    name = 'single complex matrix'
                    n = data_size // block['element_size'] * 2 * 4
                else:
                    name = 'single matrix'
                    n = data_size // block['element_size'] * 4
            else:
                if self.is_set(block['data_type'], self.DataType.Complex):
                    name = 'single complex'
                    n = data_size // block['element_size'] * 2
                else:
                    name = 'single'
                    n = data_size // block['element_size']
            data = struct.unpack('f' * n, f.read(data_size))

        elif self.is_set(block['data_type'], self.DataType.Double):
            if self.is_set(block['data_type'], self.DataType.Matrix):
                if self.is_set(block['data_type'], self.DataType.Complex):
                    name = 'double complex matrix'
                    n = data_size // block['element_size'] * 2 * 4
                else:
                    name = 'double matrix'
                    n = data_size // block['element_size'] * 4
            else:
                if self.is_set(block['data_type'], self.DataType.Complex):
                    name = 'double complex'
                    n = data_size // block['element_size'] * 2
                else:
                    name = 'double'
                    n = data_size // block['element_size']
            data = struct.unpack('d ' * n, f.read(data_size))

        else:
            raise ValueError('ERROR: Unknown binary data type detected.')

        # Add the data block into the block dictionary.
        block['data_type_name'] = name
        block['data_length'] = n
        block['data'] = numpy.squeeze(data)

        if (self.is_set(block['data_type'], self.DataType.Double) or
                self.is_set(block['data_type'], self.DataType.Single)) \
                and block['data'].shape != ():
            assert len(block['data'].shape) == 1, \
                'Unexpected Matrix like block data shape detected ' \
                '@ block number %i id:(%i.%i.%i)' % (block['number'],
                                                     block['group'],
                                                     block['tag'],
                                                     block['index'])
            # Convert complex data to python complex type
            if self.is_set(block['data_type'], self.DataType.Complex):
                block['data'] = numpy.array([complex(v[0], v[1]) for v
                                             in block['data'].reshape(n // 2, 2)
                                             ])
                block['block_length'] = n / 2
            # Wrap matrix data into 2 x 2 blocks.
            if self.is_set(block['data_type'], self.DataType.Matrix):
                n = block['block_length']
                block['data'] = block['data'].reshape(n // 4, 2, 2)

        if block['flag_crc']:
            # TODO(BM) implement CRC check. e.g. http://goo.gl/IfyyOO
            f.read(4)

    def read_data(self):
        """."""
        f = self.file_handle
        block_id = 0
        while f.read(3) == b'TBG':
            block = self.read_block_header(block_id)
            self.read_block_data(block)
            block_id += 1

    def read(self):
        """."""
        self.read_header()
        self.read_data()

    def date_time(self):
        gid = self.Group.Standard
        tid = self.Standard.DateTime
        if gid in self.record and tid in self.record[gid]:
            assert len(self.record[gid][tid]) == 1, \
                'Expecting only one standard group, date-time tag!'
            return self.record[gid][tid][0]['data']

    def user(self):
        gid = self.Group.Standard
        tid = self.Standard.UserName
        if gid in self.record and tid in self.record[gid]:
            assert len(self.record[gid][tid]) == 1, \
                'Expecting only one standard group, user tag!'
            return self.record[gid][tid][0]['data']

    def settings(self):
        gid = self.Group.Settings
        tid = self.Settings.File
        if gid in self.record and tid in self.record[gid]:
            assert len(self.record[gid][tid]) == 1, \
                'Expecting only one standard group, settings tag!'
            return self.record[gid][tid][0]['data']

    def print_summary(self):
        for group_id in self.record:
            group_data = self.record[group_id]
            for tag_id in group_data:
                tag_data = group_data[tag_id]
                for index in tag_data:
                    block = tag_data[index]
                    print('[%03i]' % block['number'], end=' ')
                    block_id = '%i.%i.%i' % (group_id, tag_id, index)
                    print('%-9s' % block_id, end=' ')
                    if block['flag_crc']:
                        print('crc', end=' ')
                    print('')


class OskarVis(OskarBinary):

    """."""

    class VisHeader:
        TelescopePath = 1
        NumVisBlockTags = 2
        FlagAutoCorrelation = 3
        FlagCrossCorrelation = 4
        VisDataType = 5
        CoordDataType = 6
        MaxTimes = 7
        NumTimes = 8
        MaxChannels = 9
        NumChannels = 10
        NumStations = 11
        PolarisationType = 12
        PhaseCentreCoordType = 21
        PhaseCentre = 22
        StartFrequency = 23
        FrequencyIncrement = 24
        ChannelBandwidth = 25
        StartTime = 26
        TimeInterval = 27
        TimeIntegration = 28
        TelescopeLon = 29
        TelescopeLat = 30
        TelescopeAlt = 31
        StationX = 32
        StationY = 33
        StationZ = 34

    class VisBlock:
        Dims = 1
        AutoCorrelation = 2
        CrossCorrelation = 3
        UU = 4
        VV = 5
        WW = 6

    class PolarisationType:
        IQUV = 0,
        I = 1,
        Q = 2,
        U = 3,
        V = 4,
        Linear = 10,
        XX = 11,
        XY = 12,
        YX = 13,
        YY = 14

    def __init__(self, file_name):

        OskarBinary.__init__(self, file_name)
        # super(OskarVis, self).print_summary()
        if not self.bin_ver == 2:
            raise ValueError("Only OSKAR binary format version-2.0 files "
                             "can be read by this class.")

        # Make local copies of visibility header variables.
        vis_header = self.record[self.Group.VisHeader]
        assert len(vis_header) == 26, \
            'Expecting the visibility header to have 26 tags!'
        self.block_length = vis_header[self.VisHeader.MaxTimes][0]['data']
        self.num_times = vis_header[self.VisHeader.NumTimes][0]['data']
        self.num_channels = vis_header[self.VisHeader.NumChannels][0]['data']
        self.num_stations = vis_header[self.VisHeader.NumStations][0]['data']
        self.num_baselines = self.num_stations * (self.num_stations - 1) // 2
        self.num_blocks = int(numpy.ceil(float(self.num_times) /
                                         self.block_length))
        self.pol_type = vis_header[self.VisHeader.PolarisationType][0]['data']
        self._start_time = vis_header[self.VisHeader.StartTime][0]['data']
        self._time_interval = vis_header[self.VisHeader.TimeInterval][0]['data']
        self._phase_centre_type = vis_header[self.VisHeader.PhaseCentreCoordType][0]['data']
        self._phase_centre = vis_header[self.VisHeader.PhaseCentre][0]['data']
        self._cross_correlation = vis_header[self.VisHeader.FlagCrossCorrelation][0]['data']
        self._auto_correlation = vis_header[self.VisHeader.FlagAutoCorrelation][0]['data']

        self.telescope_path = vis_header[self.VisHeader.TelescopePath][0]['data'].tostring().decode()[:-1]
        self.telescope_lon = vis_header[self.VisHeader.TelescopeLon][0]['data']
        self.telescope_lat = vis_header[self.VisHeader.TelescopeLat][0]['data']
        self.telescope_alt = vis_header[self.VisHeader.TelescopeAlt][0]['data']
        self.station_x = vis_header[self.VisHeader.StationX][0]['data']
        self.station_y = vis_header[self.VisHeader.StationY][0]['data']
        self.station_z = vis_header[self.VisHeader.StationZ][0]['data']

        #
        # block_dims = self.data[self.Group.VisBlock][self.VisBlock.Dims]
        # for index in block_dims:
        #     print index, block_dims[index]['data']

    def uvw(self, flatten=False):
        # FIXME(BM) handle channels?
        # FIXME(BM) uvw coordinates when auto-correlations are present.
        group = self.Group.VisBlock
        tag_uu = self.VisBlock.UU
        tag_vv = self.VisBlock.VV
        tag_ww = self.VisBlock.WW
        uu = numpy.empty((self.num_times, self.num_baselines), dtype='f8')
        vv = numpy.empty((self.num_times, self.num_baselines), dtype='f8')
        ww = numpy.empty((self.num_times, self.num_baselines), dtype='f8')
        for index in range(0, self.num_blocks):
            block_dims = self.record[group][self.VisBlock.Dims][index]['data']
            block_times = block_dims[2]
            block_time_start = block_dims[0]
            block_baselines = block_dims[4]
            assert block_baselines == self.num_baselines, \
                "Data dimension mismatch"
            assert block_times <= self.block_length, \
                "Invalid block length ?!."
            uu_block = self.record[group][tag_uu][index]['data']
            uu_block = uu_block[0:block_baselines * block_times]
            uu_block = uu_block.reshape((block_times, block_baselines))
            uu[block_time_start:block_time_start + block_times, :] = uu_block
            vv_block = self.record[group][tag_vv][index]['data']
            vv_block = vv_block[0:block_baselines * block_times]
            vv_block = vv_block.reshape((block_times, block_baselines))
            vv[block_time_start:block_time_start + block_times, :] = vv_block
            ww_block = self.record[group][tag_ww][index]['data']
            ww_block = ww_block[0:block_baselines * block_times]
            ww_block = ww_block.reshape((block_times, block_baselines))
            ww[block_time_start:block_time_start + block_times, :] = ww_block
        # FIXME(BM): The data starts flat so if flatten, just don't reshape?
        if flatten:
            uu = uu.flatten()
            vv = vv.flatten()
            ww = ww.flatten()
        return uu, vv, ww

    def amplitudes(self, flatten=False):
        group = self.Group.VisBlock
        tag = self.VisBlock.CrossCorrelation
        tag_dims = self.VisBlock.Dims

        if self.pol_type == self.PolarisationType.I:
            amp = numpy.empty((self.num_times, self.num_baselines), dtype='c16')
            for index in range(0, self.num_blocks):
                block_dims = self.record[group][tag_dims][index]['data']
                block_time_start = block_dims[0]
                block_times = block_dims[2]
                block_baselines = block_dims[4]
                assert block_baselines == self.num_baselines, \
                    "Data dimension mismatch"
                assert block_times <= self.block_length, \
                    "Invalid block length ?!."
                amp_block = self.record[group][tag][index]['data']
                amp_block = amp_block[0:block_baselines * block_times]
                amp_block = amp_block.reshape((block_times, block_baselines))
                amp[block_time_start:block_time_start + block_times, :] = \
                    amp_block
            if flatten:
                amp = amp.flatten()
            return amp

        elif self.pol_type == self.PolarisationType.Linear:
            amp = numpy.empty((self.num_times, self.num_baselines, 2, 2),
                              dtype='c16')
            for index in range(0, self.num_blocks):
                block_dims = self.record[group][tag_dims][index]['data']
                block_time_start = block_dims[0]
                block_times = block_dims[2]
                block_baselines = block_dims[4]
                assert block_baselines == self.num_baselines, \
                    "Data dimension mismatch"
                assert block_times <= self.block_length, \
                    "Invalid block length ?!."
                amp_block = self.record[group][tag][index]['data']
                amp_block = amp_block[0:block_baselines * block_times]
                amp_block = amp_block.reshape((block_times, block_baselines,
                                               2, 2))
                amp[block_time_start:block_time_start + block_times, :] = \
                    amp_block
            if flatten:
                amp = amp.reshape(self.num_baselines * self.num_times, 2, 2)
            return amp

    def stokes_i(self, flatten=True):
        amp = self.amplitudes(flatten)
        if len(amp.shape) > 1:
            if self.pol_type == self.PolarisationType.Linear:
                amp = 0.5 * (amp[:, 0, 0] + amp[:, 1, 1])
            else:
                raise ValueError('Unexpected polarisation type.')
        return amp

    def times(self, flatten=False):
        """ Returns visibility times in MDJ UTC """
        time_interval_mjd = self._time_interval / (3600.0 * 24.0)
        times = self._start_time + time_interval_mjd * numpy.arange(self.num_times)
        if flatten:
            return numpy.repeat(times, int(self.num_baselines))
        else:
            return numpy.transpose(numpy.tile(times, (int(self.num_baselines), 1)))

    def stations(self, flatten=False):
        assert self._cross_correlation, \
            "Reading non-cross-correlation data not fully supported yet!"

        # Order according to documentation is 0-1, 0-2, 0-3... 1-2, ...
        station1 = numpy.repeat(numpy.arange(self.num_stations),
                                self.num_stations-1 - numpy.arange(self.num_stations))
        station2 = numpy.hstack([numpy.arange(start+1, self.num_stations)
                                 for start in numpy.arange(self.num_stations)])
        # Tile in one or two dimensions depending on whether we want a
        # flat result
        tiles = (int(self.num_times) if flatten else [int(self.num_times), 1])
        return numpy.tile(station1, tiles), numpy.tile(station2, tiles)

    def phase_centre(self, flatten=False):
        """ Returns RA and DEC of the phase centre in degrees """
        assert self._phase_centre_type == 0, \
            "Unknown phase centre type %d!" % self._phase_centre_type
        return (self._phase_centre[0], self._phase_centre[1])

    def frequency(self, channel=0):
        group = self.Group.VisHeader
        tag = self.VisHeader.StartFrequency
        index = 0
        start_freq = self.record[group][tag][index]['data']
        tag = self.VisHeader.FrequencyIncrement
        freq_inc = self.record[group][tag][index]['data']
        return start_freq + channel * freq_inc

    def print_summary(self, verbose=False):
        print('No. times     : %i' % self.num_times)
        print('No. channels  : %i' % self.num_channels)
        print('No. baselines : %i' % self.num_baselines)
        if verbose:
            for group_id in self.record:
                group_data = self.record[group_id]
                for tag_id in group_data:
                    tag_data = group_data[tag_id]
                    for index in tag_data:
                        block = tag_data[index]
                        print('[%03i]' % block['number'], end=' ')
                        block_id = '%i.%i.%i' % (group_id, tag_id, index)
                        print('%-9s' % block_id, end=' ')
                        group_name = ''
                        if group_id == self.Group.VisHeader:
                            group_name = 'VisHeader'
                        if group_id == self.Group.VisBlock:
                            group_name = 'VisBlock'
                        print('%-15s' % group_name, end=' ')
                        if block['flag_crc']:
                            print('crc', end=' ')
                        print('')


def import_visibility_from_oskar(oskar_file: str, **kwargs) -> Visibility:
    """ Import a visibility set from an OSKAR visibility file

    :param oskar_file: Name of OSKAR visibility file
    :returns: Visibility
    """
    
    # Extract data from Oskar file
    oskar_vis = OskarVis(oskar_file)
    ra, dec = oskar_vis.phase_centre()
    a1, a2 = oskar_vis.stations(flatten=True)
    
    # Make configuration
    location = EarthLocation(lon=oskar_vis.telescope_lon,
                             lat=oskar_vis.telescope_lat,
                             height=oskar_vis.telescope_alt)
    antxyz = numpy.transpose([oskar_vis.station_x,
                              oskar_vis.station_y,
                              oskar_vis.station_z])
    config = Configuration(
        name=oskar_vis.telescope_path,
        location=location,
        xyz=antxyz
    )
    
    # Construct visibilities
    return Visibility(
        frequency=[oskar_vis.frequency(i) for i in range(oskar_vis.num_channels)],
        phasecentre=SkyCoord(frame=ICRS, ra=ra, dec=dec, unit=u.deg),
        configuration=config,
        uvw=numpy.transpose(oskar_vis.uvw(flatten=True)),
        time=oskar_vis.times(flatten=True),
        antenna1=a1,
        antenna2=a2,
        vis=oskar_vis.amplitudes(flatten=True),
        weight=numpy.ones(a1.shape))



