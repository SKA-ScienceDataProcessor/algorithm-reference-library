from setuptools import setup, Extension
from distutils.sysconfig import get_config_var
from setuptools.command.build_ext import build_ext
from subprocess import call
import os
import sys

# Bail on Python < 3
assert sys.version_info[0] >= 3


ffi_demo = Extension('ffi_demo', sources = ['src/ffi_demo.c', 'src/arlwrap.c'],
        undef_macros = ['NDEBUG'], extra_compile_args = ['-Wno-strict-prototypes'])

class CFFIBuild(build_ext):
        def run(self):
            super().run()

            cmd = [
                    "make",
                    "BUILD_LIB={}".format(self.build_lib)
                ]

            call(cmd)

setup(name='FFI_Demo',
    version='0.1',
    python_requires='>=3',
    description = "Demo for compilation of FFI-wrapped Python callable from C",
    ext_modules = [ffi_demo],
    cmdclass = {'build_ext': CFFIBuild})

