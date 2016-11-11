# -*- coding: utf-8 -*-
"""OSKAR simulation script for use with the VLA-A telescope model.

Requires:
    python (2.7.x), numpy (1.9.x)
    oskar-2.6.1 (http://www.oerc.ox.ac.uk/~ska/oskar2)
"""

import collections
import os
import numpy
import subprocess
import argparse


def create_sky_model(file_name, ra, dec, stokes_i):
    """Creates a basic OSKAR sky model.

    Converts the list of source positions supplied as arguments ra, dec and
    Stokes I flux supplied as the argument stokes_i and writes an oskar sky
    model file.

    see: http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Sky-Model.pdf
    """
    fh = open(file_name, 'w')
    for ra_, dec_, flux in zip(ra, dec, stokes_i):
        fh.write('%.14f, %.14f, %.3f\n' % (ra_, dec_, flux))
    fh.close()


def dict_to_ini(settings_dict, ini):
    """Convert a dictionary of settings to and OSKAR settings ini file.

    see: http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Settings.pdf
    """
    ini_dir = os.path.dirname(ini)
    if not ini_dir == "" and not os.path.isdir(ini_dir):
        os.makedirs(ini_dir)
    for group in sorted(settings_dict):
        for key in sorted(settings_dict[group]):
            key_ = group + key
            value_ = settings_dict[group][key]
            subprocess.call(["oskar_settings_set", "-q", ini, key_,
                             str(value_)])


def create_settings(sky_file, freq_hz, start_time_mjd, obs_length, num_times,
                    ra0, dec0, vis_name):
    """Create a dictionary of simulation settings."""
    s = collections.OrderedDict()
    s['simulator/'] = {
        'max_sources_per_chunk': 128,
        'double_precision': 'true',
        'keep_log_file': 'false'
    }
    s['sky/'] = {
        # 'oskar_sky_model/file': sky_file,
        'generator/grid/side_length': 11,
        'generator/grid/fov_deg': 3,
        'generator/grid/mean_flux_jy': 1
    }
    s['observation/'] = {
        'start_frequency_hz': freq_hz,
        'num_channels': 1,
        'start_time_utc': start_time_mjd,
        'length': obs_length,
        'num_time_steps': num_times,
        'phase_centre_ra_deg': ra0,
        'phase_centre_dec_deg': dec0
    }
    s['telescope/'] = {
        'longitude_deg': -107.6184,
        'latitude_deg': 34.0790,
        'input_directory': os.path.join('models', 'VLA_A.tm'),
        'pol_mode': 'Scalar',
        'station_type': 'Isotropic beam'
    }
    s['interferometer/'] = {
        'time_average_sec': 0.0,
        'channel_bandwidth_hz': 0.0,
        'ms_filename': vis_name + '.ms',
        'oskar_vis_filename': vis_name + '.vis'
    }
    return s


def main(output_path):
    """."""
    dtype = [('ra', 'f8'), ('dec', 'f8'), ('date-time', 'a25'),
             ('mjd', 'f8'), ('az', 'f8'), ('el', 'f8')]
    pointing = numpy.loadtxt(os.path.join('models', 'pointings_vla.txt'),
                             dtype=dtype)

    # ----------------------------------------
    pointing_idx = 0
    ra0 = pointing['ra'][pointing_idx]
    dec0 = pointing['dec'][pointing_idx]
    sky_file = os.path.join(output_path, 'test.osm')
    freq_hz = 74.0e6  # 4-band (0.058-0.084 GHz)
    start_time_mjd = pointing['mjd'][pointing_idx]
    num_times = 100
    obs_length = 6.0 * 3600.0  # seconds
    vis_name = os.path.join(output_path, 'test_vla')
    ini = os.path.join(output_path, 'test.ini')
    # ----------------------------------------

    # ra = [ra0, ra0 + 0.05, ra0 + 0.18, ra0 + 0.3, ra0, ra0, ra0]
    # dec = [dec0, dec0 + 0.05, dec0, dec0 - 0.1, dec0 + 0.25, dec0 + 0.4,
    #        dec0 + 1.0]
    # flux = numpy.ones((len(ra),), dtype='f8')
    # flux[-1] = 3.0
    # create_sky_model(sky_file, ra, dec, flux)
    s = create_settings(sky_file, freq_hz, start_time_mjd, obs_length,
                        num_times, ra0, dec0, vis_name)
    dict_to_ini(s, ini)

    subprocess.call(["oskar_sim_interferometer", ini])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OSKAR simulation script.',
                                     epilog='''
                                     Example:
                                        $ python sim_vla.py output
                                     ''')
    parser.add_argument('out_dir', help='Simulation output directory.',
                        type=str)
    args = parser.parse_args()

    output_path = args.out_dir
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    main(output_path)
