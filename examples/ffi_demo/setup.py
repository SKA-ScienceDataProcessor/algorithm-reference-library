from setuptools import setup, Extension
from distutils.sysconfig import get_config_var
from setuptools.command.build_ext import build_ext
from subprocess import call
import os
import sys

# Bail on Python < 3
assert sys.version_info[0] >= 3


# NB. These are not really Python extensions (i.e., they do not
# Py_Initialize() and they do define main() ), we are just cheating to
# re-use the setuptools build support.

ffi_demo = Extension('ffi_demo', sources = ['src/ffi_demo.c', 'src/arlwrap.c', 'src/wrap_support.c'],
        undef_macros = ['NDEBUG'], extra_compile_args = ['-Wno-strict-prototypes'])

starpu_timg = Extension('timg_starpu', sources = ['starpu_timg/timg_starpu.c',
    'starpu_timg/timg_pu_routines.c', 'src/arlwrap.c', 'src/wrap_support.c'],
        include_dirs = ['{home}/.local/starpu/include/starpu/1.2'.format(home=os.getenv('HOME'))],
        undef_macros = ['NDEBUG'], extra_compile_args = ['-Wno-strict-prototypes'])

libarlffi = Extension('libarlffi',
                   sources = ['src/arlwrap.c', 'src/wrap_support.c'],
                   undef_macros = ['NDEBUG'],
                   extra_compile_args = ['-Wno-strict-prototypes'],
                   libraries= ['cfitsio'],
)


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
    ext_modules = [ffi_demo, starpu_timg, libarlffi],
    cmdclass = {'build_ext': CFFIBuild})

