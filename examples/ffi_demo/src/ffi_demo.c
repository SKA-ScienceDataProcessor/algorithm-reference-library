#include <Python.h>
#include <stdio.h>

#include "arlwrap.h"


/* Simple exit-on-error */
void pycheck(PyObject *obj)
{
	if (!obj) {
		PyErr_Print();
		exit(1);
	}
}

/* In: module name, function name
 * Out: function address */
void *get_ffi_fn_addr(const char* module, const char* fn_name)
{
	PyObject *mod, *fn, *fn_addr;

	pycheck(mod = PyImport_ImportModule(module));
	pycheck(fn = PyObject_GetAttrString(mod, fn_name));
	pycheck(fn_addr = PyObject_GetAttrString(fn, "address"));

	return (void*)PyNumber_AsSsize_t(fn_addr, NULL);
}

/* DO NOT USE - we do not want PyObjects */
/* Leaving for reference only */
PyObject *get_plain_fn_addr(const char* module, const char* fn_name)
{
	PyObject *mod, *fn;

	pycheck(mod = PyImport_ImportModule(module));
	pycheck(fn = PyObject_GetAttrString(mod, fn_name));

	return fn;
}


int main(int argc, char **argv)
{
	Py_Initialize();

	ARLVis *vin = malloc(sizeof(ARLVis));
	ARLVis *vout = malloc(sizeof(ARLVis));

	vin->nvis = 1;
	vin->npol = 4;

	// malloc to ARLDataVisSize
	vin->data = malloc(72+(32*vin->npol*vin->nvis) * sizeof(char));
	vout->data = malloc(72+(32*vin->npol*vin->nvis) * sizeof(char));

	if (!vin->data || !vout->data) {
		fprintf(stderr, "Malloc error\n");
		exit(1);
	}

	arl_copy_visibility(vin, vout, false);

	return 0;
}
