#!/usr/bin/env python

from setuptools import setup, Extension
import os
import numpy

gridmodule = Extension('arl.core.c',
                       include_dirs = [numpy.get_include()],
                       sources=['arl/core/cmodule.c'])
setup(name='algorithm-reference-library',
      version='0.7',
      description='Algorithm Reference Library for Radio Interferometry',
      long_description=open('README.md').read(),
      author='Tim Cornwell, Peter Wortmann, Bojan Nikolic',
      author_email='realtimcornwell@gmail.com',
      url='https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library',
      license='Apache License Version 2.0',
      packages=['arl', 'examples', 'tests'],
      test_suite="tests",
      tests_require=['pytest']
      )
