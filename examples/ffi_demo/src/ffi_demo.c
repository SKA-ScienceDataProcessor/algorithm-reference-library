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

/* Temporary routine, read pickled WCS object from file
 * - only necessary while earlier stages of pipeline are not implemented */
void copy_wcs(Image *im)
{
	int len;
	FILE *f = fopen("wcs.pickle", "r");
	fseek(f,0,SEEK_END);
	len = ftell(f);
	rewind(f);
	printf("len: %d\n", len);
	fread(im->wcs, len, 1, f);
	fclose(f);
}

/* Temporary routine, read pickled polarisation frame object from file
 * - only necessary while earlier stages of pipeline are not implemented */
void copy_polframe(Image *im)
{
	int len;
	FILE *f = fopen("frame.pickle", "r");
	fseek(f,0,SEEK_END);
	len = ftell(f);
	rewind(f);
	printf("len: %d\n", len);
	fread(im->polarisation_frame, len, 1, f);
	fclose(f);
}

int main(int argc, char **argv)
{
	int i;

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

	Image *model = malloc(sizeof(Image));
	Image *psf = malloc(sizeof(Image));
	Image *residual = malloc(sizeof(Image));
	Image *restored = malloc(sizeof(Image));

	model->size = 4096;
	psf->size = 4096;
	residual->size = 4096;
	restored->size = 4096;

	for(i=0; i<4; i++) {
		model->data_shape[i] = 8;
		psf->data_shape[i] = 8;
		residual->data_shape[i] = 8;
		restored->data_shape[i] = 8;
	}

	model->data = calloc(4096,sizeof(double));
	psf->data = calloc(4096,sizeof(double));
	residual->data = calloc(4096,sizeof(double));
	restored->data = calloc(4096,sizeof(double));

	model->wcs = calloc(2998,sizeof(char));
	psf->wcs = model->wcs;
	residual->wcs = model->wcs;
	restored->wcs = model->wcs;

	model->polarisation_frame = calloc(512,sizeof(char));
	psf->polarisation_frame = model->polarisation_frame;
	residual->polarisation_frame = model->polarisation_frame;
	restored->polarisation_frame = model->polarisation_frame;

	copy_wcs(model);
	copy_polframe(model);
	arl_predict_2d(vin, psf, vout);

	arl_create_image_from_visibility(vout, model);


	printf("before: %p->%f | %p->%f\n", model->data,
			((double*)(model->data))[0], restored->data,
			((double*)(restored->data))[0]);
	arl_deconvolve_cube(model, psf, restored, residual);
	printf("aftore: %p->%f | %p->%f\n", model->data,
			((double*)(model->data))[0], restored->data,
			((double*)(restored->data))[0]);
	arl_restore_cube(model, psf, residual, restored);
	printf("aftore: %p->%f | %p->%f\n", model->data,
			((double*)(model->data))[0], restored->data,
			((double*)(restored->data))[0]);


	return 0;
}
