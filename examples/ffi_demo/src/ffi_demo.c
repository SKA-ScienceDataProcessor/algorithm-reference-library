#include <Python.h>
#include <stdio.h>
#include <assert.h>

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

/*
 * Verifies that:
 * - vin and vout are unique in memory
 * - vin and vout have equivalent values
 */
int verify_arl_copy(ARLVis *vin, ARLVis *vout)
{
	char *vindata_bytes, *voutdata_bytes;
	int ARLVisDataSize;
	int i;

	if (vin == vout) {
		fprintf(stderr, "vin == vout\n");
		return 1;
	}

	if (!((vin->nvis == vout->nvis) && (vin->npol == vout->npol))) {
		return 2;
	}

	if (vin->data == vout->data) {
		return 3;
	}

	ARLVisDataSize = 72 + (32 * vin->npol * vin->nvis);
	vindata_bytes = (char*) vin->data;
	voutdata_bytes = (char*) vout->data;

	for (i=0; i<ARLVisDataSize; i++) {
		if (vindata_bytes[i] != voutdata_bytes[i]) {
			return 4;
		}
	}

	return 0;
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

	((ARLVisEntryP4 *)(vin->data))[0].time=99;

	if (!vin->data || !vout->data) {
		fprintf(stderr, "Malloc error\n");
		exit(1);
	}

	arl_copy_visibility(vin, vout, false);

	assert(0 == verify_arl_copy(vin, vout));

	return 0;
}
