#!/usr/bin/env python

from distutils.core import setup

setup(name='crocodile',
      version='0.2',
      description='Algorithm Reference Library for Radio Interferometry',
      long_description=open('README.md').read(),
      author='Tim Cornwell, Bojan Nikolic, Peter Wortmann',
      author_email='realtimcornwell@gmail.com',
      url='https://github.com/SKA-ScienceDataProcessor/crocodile',
      license='Apache License Version 2.0',
      packages=['crocodile', 'arl', 'examples', 'tests'],
      )
