#!/usr/bin/env python

#from setuptools import setup
from setuptools import setup, Extension
from distutils.sysconfig import get_config_var, get_config_vars
from distutils.spawn import find_executable
from setuptools.command.build_ext import build_ext
from subprocess import call
import subprocess
import os
import sys
import pkgconfig
import shutil

# Bail on Python < 3
assert sys.version_info[0] >= 3


# MF. This is a workaround to be able to build the library with MacOS
if sys.platform == 'darwin':
	vars = get_config_vars()
	vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle','-dynamiclib')
	os.environ["CC"] = "clang"

if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = find_executable("protoc")

if 'PROTOC_C' in os.environ and os.path.exists(os.environ['PROTOC_C']):
  protocc = os.environ['PROTOC_C']
else:
  protocc = find_executable("protoc-c")

def generate_proto(source, srcdir):
  """Invokes the Protocol Compiler to generate a _pb2.py from the given
  .proto file.  Does nothing if the output already exists and is newer than
  the input."""

  output = source.replace(".proto", "_pb2.py").replace("../src/", "")

  if (not os.path.exists(output) or
      (os.path.exists(source) and
       os.path.getmtime(source) > os.path.getmtime(output))):
    print("Generating %s..." % output)

    if not os.path.exists(srcdir + '/' + source):
      sys.stderr.write("Can't find required file: %s\n" % source)
      sys.exit(-1)

    if protoc == None:
      sys.stderr.write(
          "protoc is not installed nor found in ../src.  Please compile it "
          "or install the binary package.\n")
      sys.exit(-1)

    protoc_command = [ protoc, "-I./", "--python_out=.", source ]
    if subprocess.call(protoc_command, cwd=srcdir) != 0:
      sys.exit(-1)
    protocc_command = [ protocc, "-I.",  "--c_out=.", source ]
    if subprocess.call(protocc_command, cwd=srcdir) != 0:
      sys.exit(-1)

generate_proto('arl.proto', 'ffiwrappers/src')
shutil.copy('ffiwrappers/src/arl.pb-c.h','ffiwrappers/include/arl.pb-c.h')
# NB. These are not really Python extensions (i.e., they do not
# Py_Initialize() and they do define main() ), we are just cheating to
# re-use the setuptools build support.

arlproto = Extension('arlproto',
		   sources = ['ffiwrappers/src/arl.proto']
		   
)

libarlffi = Extension('libarlffi',
                   sources = ['ffiwrappers/src/arlwrap.c', 'ffiwrappers/src/wrap_support.c', 'ffiwrappers/src/wrappingcore.c', 'ffiwrappers/src/arl.pb-c.c'],
                   include_dirs=['/usr/include/cfitsio'],
                   undef_macros = ['NDEBUG'],
                   extra_compile_args = ['-Wno-strict-prototypes', pkgconfig.cflags('libprotobuf-c'), pkgconfig.cflags('cfitsio') ],
                   libraries = ['cfitsio'],
		   optional=True,
)

setup(name='algorithm-reference-library',
      version='0.9',
      python_requires='>=3',
      description='Algorithm Reference Library for Radio Interferometry',
      long_description=open('README.md').read(),
      author='Tim Cornwell, Peter Wortmann, Bojan Nikolic',
      author_email='realtimcornwell@gmail.com',
      url='https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library',
      license='Apache License Version 2.0',
      packages=['data_models', 'processing_library', 'processing_components', 'workflows', 'examples'],
      test_suite="tests",
      tests_require=['pytest'],
      ext_modules = [libarlffi]
      )




class CFFIBuild(build_ext):
        def run(self):
            super().run()

            cmd = [
                    "make",
                    "BUILD_LIB={}".format(self.build_lib)
                ]

            call(cmd)

#setup(name='FFI_Demo',
#    version='0.1',
#    python_requires='>=3',
#    description = "Demo for compilation of FFI-wrapped Python callable from C",
#    ext_modules = [libarlffi])
#,
#    cmdclass = {'build_ext': CFFIBuild})

