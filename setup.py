#!/usr/bin/env python

from setuptools import setup

setup(name='algorithm-reference-library',
      version='0.9',
      description='Algorithm Reference Library for Radio Interferometry',
      long_description=open('README.md').read(),
      author='Tim Cornwell, Peter Wortmann, Bojan Nikolic',
      author_email='realtimcornwell@gmail.com',
      url='https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library',
      license='Apache License Version 2.0',
      packages=['data_models', 'libs', 'processing_components', 'workflows', 'examples', 'libs', 'data'],
      test_suite="libs",
      tests_require=['pytest']
      )
