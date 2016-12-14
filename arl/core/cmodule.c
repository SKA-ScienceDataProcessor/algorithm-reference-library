
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <modsupport.h>
#include <numpy/arrayobject.h>

#include <complex.h>

static PyObject *
core_gridder(PyObject *dummy, PyObject *args)
{
    PyObject *kernel_arg=NULL, *uvgrid_arg=NULL, *vis_arg=NULL, *xy_arg[2] = {NULL,NULL}, *kernel_ixs_arg=NULL;
    PyArrayObject *kernel=NULL, *uvgrid=NULL, *vis=NULL, *xy[2] = {NULL, NULL}, *kernel_ixs=NULL;

    // Parse parameters
    if (!PyArg_ParseTuple(args, "O!OOO|OO",
                          &PyArray_Type, &uvgrid_arg,
                          &vis_arg,
                          &xy_arg[0], &xy_arg[1],
                          &kernel_arg,
                          &kernel_ixs_arg)) return NULL;

    // Get arrays
    uvgrid = (PyArrayObject *)PyArray_FROM_OTF(uvgrid_arg, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY);
    if (uvgrid == NULL) goto fail;
    vis = (PyArrayObject *)PyArray_FROM_OTF(vis_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    xy[0] = (PyArrayObject *)PyArray_FROM_OTF(xy_arg[0], NPY_INT64, NPY_ARRAY_IN_ARRAY);
    xy[1] = (PyArrayObject *)PyArray_FROM_OTF(xy_arg[1], NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (vis == NULL || xy[0] == NULL || xy[1] == NULL) goto fail;
    if (kernel_arg != NULL && kernel_arg  != Py_None) {
        kernel = (PyArrayObject *)PyArray_FROM_OTF(kernel_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        if (kernel == NULL) return NULL;
    }
    if (kernel_ixs_arg != NULL && kernel_ixs_arg != Py_None) {
        kernel_ixs = (PyArrayObject *)PyArray_FROM_OTF(kernel_ixs_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        if (kernel_ixs == NULL) goto fail;
    }

    // Check that dimensions UV-grid format is as expected
    if (PyArray_NDIM(uvgrid) != 2) {
        PyErr_SetString(PyExc_ValueError, "UV grid must have two dimensions");
        goto fail;
    }
    npy_intp grid_hgt = PyArray_DIM(uvgrid, 0);
    npy_intp grid_wdt = PyArray_DIM(uvgrid, 1);

    // Check dimension
    if (PyArray_NDIM(vis) != 1) {
        PyErr_SetString(PyExc_ValueError, "Visibilities must have one dimension");
        goto fail;
    }
    if (PyArray_NDIM(xy[0]) != 1) {
        PyErr_SetString(PyExc_ValueError, "X coordinates must have one dimension");
        goto fail;
    }
    if (PyArray_NDIM(xy[1]) != 1) {
        PyErr_SetString(PyExc_ValueError, "Y coordinates must have one dimension");
        goto fail;
    }
    npy_intp vis_count = PyArray_DIM(vis, 0);
    if (PyArray_DIM(xy[0], 0) != vis_count || PyArray_DIM(xy[1], 0) != vis_count) {
        char message[128];
        sprintf(message, "Visibility count mismatch (vis %ld, X %ld, Y %ld)",
                vis_count, PyArray_DIM(xy[0], 0), PyArray_DIM(xy[1], 0));
        PyErr_SetString(PyExc_ValueError, message);
        goto fail;
    }

    // Check kernel parameter, determine number of indices
    npy_intp ix_count = 0;
    npy_intp kern_hgt = 1;
    npy_intp kern_wdt = 1;
    if (kernel) {
        if (PyArray_NDIM(kernel) < 2) {
            PyErr_SetString(PyExc_ValueError, "Kernels must have at least two dimensions");
            goto fail;
        }
        // All remaining dimensions must be kernel indices
        ix_count = PyArray_NDIM(kernel) - 2;
        kern_hgt = PyArray_DIM(kernel, ix_count);
        kern_wdt = PyArray_DIM(kernel, ix_count+1);
    }

    // Check that we've got enough indices
    npy_intp ix_got = 0;
    if (!kernel_ixs || PyArray_NDIM(kernel_ixs) == 0) {
        ix_got = 0;
    } else if (PyArray_NDIM(kernel_ixs) == 1) {
        ix_got = 1;
    } else if (PyArray_NDIM(kernel_ixs) == 2) {
        ix_got = PyArray_DIM(kernel_ixs, 1);
    } else {
        PyErr_SetString(PyExc_ValueError, "Kernel indices must have two dimensions maximum");
        goto fail;
    }
    if (ix_count != ix_got) {
        char message[128];
        sprintf(message, "Must pass enough kernel indices to determine kernel to use (got %ld, need %ld)",
                ix_got, ix_count);
        PyErr_SetString(PyExc_ValueError, message);
        goto fail;
    }
    if (kernel_ixs && PyArray_DIM(kernel_ixs, 0) != vis_count) {
        char message[128];
        sprintf(message, "Visibility count mismatch (vis %ld, kernel %ld)",
                vis_count, PyArray_DIM(kernel_ixs, 0));
        PyErr_SetString(PyExc_ValueError, message);
        goto fail;
    }

    // Gridding loop
    npy_intp *kern_ix = alloca(sizeof(npy_intp) * (ix_count + 2));
    npy_intp iv;
    for (iv = 0; iv < vis_count; iv++) {

        // Get kernel index
        npy_intp i;
        if (ix_count == 1) {
            kern_ix[0] = *(int64_t *)PyArray_GETPTR1(kernel_ixs, iv);
        } else {
            for (i = 0; i < ix_count; i++) {
                kern_ix[i] = *(int64_t *)PyArray_GETPTR2(kernel_ixs, iv, i);
            }
        }

        // Check index bounds
        for (i = 0; i < ix_count; i++) {
            if (kern_ix[i] < 0 || kern_ix[i] >= PyArray_DIM(kernel, i)) {
                char message[128];
                sprintf(message, "Visibility %ld kernel index %ld out of bounds (%ld, max %ld)",
                        iv, i, kern_ix[i], PyArray_DIM(kernel, i));
                PyErr_SetString(PyExc_ValueError, message);
                goto fail;
            }
        }

        // Get position
        int64_t x0 = *(int64_t *)PyArray_GETPTR1(xy[0], iv);
        int64_t y0 = *(int64_t *)PyArray_GETPTR1(xy[1], iv);
        if (x0 < 0 || y0 < 0 ||
            x0 + kern_wdt > grid_wdt || y0 + kern_hgt > grid_hgt) {

            char message[128];
            sprintf(message, "Visibility %ld position out of bounds (%ld/%ld)", iv, x0, y0);
            PyErr_SetString(PyExc_ValueError, message);
            goto fail;
        }

        // Access visibility
        double complex v = *(double complex *)PyArray_GETPTR1(vis, iv);

        // Grid
        npy_intp x, y;
        for (y = 0; y < kern_hgt; y++) {
            for (x = 0; x < kern_wdt; x++) {

                // Access kernel
                kern_ix[ix_count] = y;
                kern_ix[ix_count+1] = x;
                double complex kern = kernel ? *(double complex *)PyArray_GetPtr(kernel, kern_ix) : 1.0;

                // Access grid
                double complex *pgrid = (double complex *)PyArray_GETPTR2(uvgrid, y0+y, x0+x);

                // Update
                *pgrid += v * kern;
            }
        }
    }

    // Decrease reference counts
    Py_DECREF(uvgrid);
    Py_DECREF(vis);
    Py_DECREF(xy_arg[0]);
    Py_DECREF(xy_arg[1]);
    if (kernel) { Py_DECREF(kernel); }
    if (kernel_ixs) { Py_DECREF(kernel_ixs); }

    // Return "None"
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    if (uvgrid) { PyArray_XDECREF_ERR(uvgrid); }
    if (vis) { Py_DECREF(vis); }
    if (xy[0]) { Py_DECREF(xy[0]); }
    if (xy[1]) { Py_DECREF(xy[1]); }
    if (kernel) { Py_DECREF(kernel); }
    if (kernel_ixs) { Py_DECREF(kernel_ixs); }
    return NULL;
}

const char gridder_doc[] =
    "gridder(grid, vis, xs, ys[, kernel, kernel_ixs])\n\n"
    "Grids visibilities at given positions. Convolution kernels are selected per\n"
    "visibility using ``kernel_ixs``.\n\n"
    ":param uvgrid: Grid to update (two-dimensional :class:`complex` array)\n"
    ":param vis: Visibility values (one-dimensional :class:`complex` array)\n"
    ":param xs: Visibility position (one-dimensional :class:`int` array)\n"
    ":param ys: Visibility values (one-dimensional :class:`int` array)\n"
    ":param kernel: Convolution kernel (minimum two-dimensional :class:`complex` array).\n"
    "  If the kernel has more than two dimensions, additional indices must be passed\n"
    "  in ``kernel_ixs``. Default: Fixed one-pixel kernel with value 1.\n"
    ":param kernel_ixs: Map of visibilities to kernel indices (maximum two-dimensional :class:`int` array).\n"
    "  Can be omitted if ``kernel`` requires no indices, and can be one-dimensional\n"
    "  if only one index is needed to identify kernels.\n";

static PyMethodDef CoreMethods[] = {
    { .ml_name  = "gridder",
      .ml_meth  = core_gridder,
      .ml_flags = METH_VARARGS,
      .ml_doc   = gridder_doc
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef corecmodule = {
   .m_base    = PyModuleDef_HEAD_INIT,
   .m_name    = "arl.core.c",
   .m_doc     = NULL,
   .m_size    = 0,
   .m_methods = CoreMethods
};

PyMODINIT_FUNC
PyInit_c(void)
{

    // Create module
    PyObject *m = PyModule_Create(&corecmodule);
    if (m == NULL)
        return NULL;

    // Initialize NumPy API
    import_array();
    return m;
}
