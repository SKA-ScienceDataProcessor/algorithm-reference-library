"""
Toy code that exposes some Python routines to C through CFFI
Based on https://www.mrao.cam.ac.uk/~bn204/publications/2017/2017-01-breakin.pdf
"""

import collections
import cffi

import arlwrap


ff = cffi.FFI()

# Create a new object type containing C-readable function address
ffi_func = collections.namedtuple("CFFI_func", "address")

count = 0

# ff.callback exports the C prototype
@ff.callback("int(char *)")
def print_hello_from_python(name=None):
    "Print a string from C"
    if not name:
        name = "nameless bastard"
    elif isinstance(name, ff.CData):
        name = ff.string(name).decode("utf-8")
    print("Hello from Python, {}!".format(name))

    if name:
        return len(name)
    return 0

# Store the function pointer in an CFFI_func object.
ffi_phfp = ffi_func(int(ff.cast("size_t", print_hello_from_python)))

@ff.callback("int(void)")
def get_count_from_python(name=None):
    global count
    count = count + 1;
    return count


ffi_gcfp = ffi_func(int(ff.cast("size_t", get_count_from_python)))

