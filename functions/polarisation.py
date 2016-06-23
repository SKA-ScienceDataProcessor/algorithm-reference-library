# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#


class MKernel():
    """ Mueller kernel with numpy.array, antenna1, antenna2, time
    """


class Jones():
    """ Jones kernel with numpy.array, antenna1, antenna2, time
    """


if __name__ == '__main__':
    import os

    print(os.getcwd())
