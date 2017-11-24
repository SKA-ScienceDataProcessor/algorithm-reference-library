from setuptools import setup, Extension
from distutils.sysconfig import get_config_var
from setuptools.command.build_ext import build_ext
from subprocess import call
import os

ffi_demo = Extension('ffi_demo', sources = ['src/ffi_demo.c'])

class CFFIBuild(build_ext):
        def run(self):
            super().run()

            cmd = [
                    "make",
                    "BUILD_LIB={}".format(self.build_lib),
                    "EXT_SUFFIX={}".format(get_config_var('EXT_SUFFIX'))
                ]

            call(cmd)

setup(name='FFI_Demo',
    version='0.1',
    description = "Demo for compilation of FFI-wrapped Python callable from C",
    ext_modules = [ffi_demo],
    cmdclass = {'build_ext': CFFIBuild})

