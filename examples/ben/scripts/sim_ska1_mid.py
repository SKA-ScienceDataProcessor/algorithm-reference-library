# -*- coding: utf-8 -*-
"""OSKAR simulation script for use with the combined SKA1 mid, July15, layout.

Requires:
    python (2.7.x), numpy (1.9.x)
    oskar-2.6.1 (http://www.oerc.ox.ac.uk/~ska/oskar2)
"""

import collections
import os
import numpy as np
import subprocess
import argparse


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


def create_settings(freq_hz, start_time_mjd, obs_length, num_times, ra0,
                    dec0, vis_name):
    """."""
    s = collections.OrderedDict()
    s['simulator/'] = {
        'max_sources_per_chunk': 128,
        'double_precision': 'true',
        'keep_log_file': 'false'
    }
    s['sky/'] = {
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
        'longitude_deg': 21.4429090,
        'latitude_deg': -30.7394750,
        'input_directory': os.path.join('models', 'ska1_mid.tm'),
        'pol_mode': 'Full',
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
    pointing = np.loadtxt(os.path.join('models', 'pointings_ska1_mid.txt'),
                          dtype=dtype)

    # ----------------------------------------
    pointing_idx = 0
    ra0 = pointing['ra'][pointing_idx]
    dec0 = pointing['dec'][pointing_idx]
    freq_hz = 700.0e6  # band-1
    start_time_mjd = pointing['mjd'][pointing_idx]
    num_times = 100
    obs_length = 6.0 * 3600.0  # seconds
    vis_name = os.path.join(output_path, 'test_ska1_mid')
    ini = os.path.join(output_path, 'test_ska1_mid.ini')
    # ----------------------------------------

    s = create_settings(freq_hz, start_time_mjd, obs_length, num_times,
                        ra0, dec0, vis_name)
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

