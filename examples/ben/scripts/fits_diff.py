# -*- coding: utf-8 -*-
"""Module to difference two FITS images."""

import pyfits
import sys
import os
import numpy as np
import argparse


def save_fits_image(filename, data, header, img1=None, img2=None):
    """Save a FITS image."""

    # Reshape to add the frequency axis
    data = np.reshape(data, (1, 1, data.shape[0], data.shape[1]))
    new_hdr = pyfits.header.Header()
    for i, item in enumerate(header.items()):
        if item[0] != 'HISTORY':
            new_hdr.append(item)
    if img1 and img2:
        new_hdr.append(('HISTORY', '' * 60))
        new_hdr.append(('HISTORY', '-' * 60))
        new_hdr.append(('HISTORY', 'Diff created from image1 - image2:'))
        new_hdr.append(('HISTORY', '- image1 : %s' % img1))
        new_hdr.append(('HISTORY', '- image2 : %s' % img2))
        new_hdr.append(('HISTORY', '-' * 60))
        new_hdr.append(('HISTORY', '' * 60))

    if os.path.exists(filename):
        print '+ WARNING, output FITS file already exists, overwriting.'
        os.remove(filename)
    pyfits.writeto(filename, data, new_hdr)


def load_fits_image(filename):
    """Load a FITS image."""
    hdu_list = pyfits.open(filename)
    data = hdu_list[0].data
    hdr = hdu_list[0].header
    return np.squeeze(data), hdr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Difference FITS images.'
                                                 'Performs the difference, '
                                                 'diff = file1 - file2.')
    parser.format_help()
    parser.add_argument("file1", help="FITS image 1",
                        type=str)
    parser.add_argument("file2", help="FITS image 2",
                        type=str)
    parser.add_argument('--out_name', '-o',
                        help='Output diff FITS image. (default: diff.fits)',
                        default='diff.fits',
                        type=str)
    args = parser.parse_args()

    out_name = args.out_name
    file1 = args.file1
    file2 = args.file2

    f1, h1 = load_fits_image(file1)
    f2, h2 = load_fits_image(file2)

    diff = f1 - f2

    print '-' * 80
    print '+ Image size  : %i x %i' % (f1.shape[0], f1.shape[1])
    print '+ File 1      : %s' % file1
    print '+ File 2      : %s' % file2
    print '+ Diff        : file1 - file2'
    print '+ Output name : %s' % out_name
    print '+ Diff stats:'
    print '  - Max       : % .3e' % np.max(diff)
    print '  - Min       : % .3e' % np.min(diff)
    print '  - Mean      : % .3e' % np.mean(diff)
    print '  - STD       : % .3e' % np.std(diff)
    print '  - RMS       : % .3e' % np.sqrt(np.mean(diff**2))
    print '-' * 80

    save_fits_image(out_name, diff, h1, file1, file2)
