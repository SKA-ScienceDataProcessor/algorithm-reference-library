#include <Python.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cfitsio/fitsio.h>

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

/* Export image to FITS */
/* Assuming nx*ny*nfreq */
/* ToDo - add polarization and wcs */
int export_image_to_fits_c(Image *im, char * filename) {
	int status = 0, exists;
	fitsfile *fptr;       /* pointer to the FITS file; defined in fitsio.h */
    	long  fpixel = 1, naxis = 3, nelements;
    	long naxes[3];
	naxes[0] = im->data_shape[3];
	naxes[1] = im->data_shape[2];
	naxes[2] = im->data_shape[1];

	fits_file_exists(filename, &exists, &status); /* check if the file exists */
	if(exists != 0) 
		fits_open_file(&fptr, filename, READWRITE, &status); /* open existed file */
	else
    		fits_create_file(&fptr, filename, &status);   /* create new file */
    	/* Create the primary array image  */
    	fits_create_img(fptr, DOUBLE_IMG, naxis, naxes, &status);
    	nelements = naxes[0] * naxes[1] * naxes[2];          /* number of pixels to write */
    	/* Write the array of integers to the image */
    	fits_write_img(fptr, TDOUBLE, fpixel, nelements, im->data, &status);
    	fits_close_file(fptr, &status);            /* close the file */
    	fits_report_error(stderr, status);  /* print out any error messages */
	return status;
}


int main(int argc, char **argv)
{
	int i;
	int *shape = malloc(4*sizeof(int));
	int serial_shape;
	int status;
	int nvis=1;

	double *times = calloc(1,sizeof(double));
	double *freq = malloc(1*sizeof(double));
	double *channel_bandwidth = malloc(1*sizeof(double));
	freq[0] = 1e8;
	channel_bandwidth[0] = 1e6;
	double cellsize = 0.0005;
	char config_name[] = "LOWBD2-CORE";

	ARLVis *vin = malloc(sizeof(ARLVis));
	ARLVis *vout = malloc(sizeof(ARLVis));

	ARLConf *lowconfig = malloc(sizeof(ARLConf));

	ant_t nb;

	Py_Initialize();

	// Find out the number of the antennas and the baselines, keep in nb structure
	nb.nbases = 1;
	helper_get_nbases(config_name, &nb);
	// Assigning configuraion values
	lowconfig->confname = config_name;
	lowconfig->pc_ra = 15.0;
	lowconfig->pc_dec = -45.0;
	lowconfig->times = times;
	lowconfig->ntimes = 1;
	lowconfig->freqs = freq;	
	lowconfig->nfreqs = 1;	
	lowconfig->channel_bandwidth = channel_bandwidth;	
	lowconfig->nchanwidth = 1;
	lowconfig->nbases = nb.nbases;
	lowconfig->npol = 1;
	// Find out the number of visibilities
	nvis = (lowconfig->nbases)*(lowconfig->nfreqs)*(lowconfig->ntimes);
	printf("nvis = %d\n", nvis);

//	vin->nvis = 13695;
//	vin->npol = 1;
	vin->nvis = nvis;
	vin->npol = lowconfig->npol;

	// malloc to ARLDataVisSize
	vin->data = malloc(72+4*(32*vin->npol*vin->nvis) * sizeof(char));
	vout->data = malloc(72+4*(32*vin->npol*vin->nvis) * sizeof(char));
//	vin->data = malloc(14000000);//13695
//	vout->data = malloc(14000000);//13695
	vin->pickle = malloc(72+4*(32*vin->npol*vin->nvis) * sizeof(char));
	vout->pickle = malloc(72+4*(32*vin->npol*vin->nvis) * sizeof(char));
//	vin->pickle = malloc(15000000);
//	vout->pickle = malloc(15000000);

	//((ARLVisEntryP4 *)(vin->data))[0].time=99;

	//if (!vin->data || !vout->data) {
	//	fprintf(stderr, "Malloc error\n");
	//	exit(1);
	//}

	//arl_copy_visibility(vin, vout, false);

	//assert(0 == verify_arl_copy(vin, vout));

	helper_get_image_shape(freq, cellsize, shape);
	serial_shape = shape[0] * shape[1] * shape[2] * shape[3];
	printf("Shape :%d x %d x %d x %d\n", shape[0],shape[1],shape[2],shape[3]);	
	printf("%d\n", serial_shape);

	Image *model = malloc(sizeof(Image));
	Image *m31image = malloc(sizeof(Image));
	Image *dirty = malloc(sizeof(Image));
	Image *psf = malloc(sizeof(Image));
	Image *comp = malloc(sizeof(Image));
	Image *residual = malloc(sizeof(Image));
	Image *restored = malloc(sizeof(Image));

	model->size = serial_shape;
	m31image->size = serial_shape;
	dirty->size = serial_shape;
	psf->size = serial_shape;
	comp->size = serial_shape;
	residual->size = serial_shape;
	restored->size = serial_shape;

	for(i=0; i<4; i++) {
		model->data_shape[i] = shape[i];
		m31image->data_shape[i] = shape[i];
		dirty->data_shape[i] = shape[i];
		psf->data_shape[i] = shape[i];
		comp->data_shape[i] = shape[i];
		residual->data_shape[i] = shape[i];
		restored->data_shape[i] = shape[i];
	}

	model->data = calloc(serial_shape,sizeof(double));
	m31image->data = calloc(serial_shape,sizeof(double));
	dirty->data = calloc(serial_shape,sizeof(double));
	psf->data = calloc(serial_shape,sizeof(double));
	comp->data = calloc(serial_shape,sizeof(double));
	residual->data = calloc(serial_shape,sizeof(double));
	restored->data = calloc(serial_shape,sizeof(double));

	model->wcs = calloc(2998,sizeof(char));
	m31image->wcs = calloc(2998,sizeof(char));
	dirty->wcs = calloc(2998,sizeof(char));
	psf->wcs = calloc(2996,sizeof(char));
	comp->wcs = calloc(2996,sizeof(char));
	residual->wcs = calloc(2996,sizeof(char));
	restored->wcs = calloc(2996,sizeof(char));

	model->polarisation_frame = calloc(512,sizeof(char));
	m31image->polarisation_frame = calloc(512,sizeof(char));
	dirty->polarisation_frame = calloc(512,sizeof(char));
	psf->polarisation_frame = calloc(512,sizeof(char));
	comp->polarisation_frame = calloc(512,sizeof(char));
	residual->polarisation_frame = calloc(512,sizeof(char));
	restored->polarisation_frame = calloc(512,sizeof(char));

//	arl_create_visibility("LOWBD2-CORE", times, freq, channel_bandwidth, vin);
	arl_create_visibility1(lowconfig, vin);

	arl_create_test_image(freq, cellsize, m31image);

	////copy_wcs(model);
	////copy_polframe(model);
	arl_predict_2d(vin, m31image, vout);

	ARLVis *tmp = vout;
	vout = vin;
	vin = tmp;


	arl_create_image_from_visibility(vin, model);

	double *sumwt = malloc(sizeof(double));

	arl_invert_2d(vin, model, false, dirty, sumwt);
	arl_invert_2d(vin, model, true, psf, sumwt);


	//printf("before: %p->%f | %p->%f\n", model->data,
	//		((double*)(model->data))[0], restored->data,
	//		((double*)(restored->data))[0]);
	arl_deconvolve_cube(dirty, psf, comp, residual);
	//printf("aftore: %p->%f | %p->%f\n", model->data,
	//		((double*)(model->data))[0], restored->data,
	//		((double*)(restored->data))[0]);
	arl_restore_cube(comp, psf, residual, restored);
	//printf("aftore: %p->%f | %p->%f\n", model->data,
	//		((double*)(model->data))[0], restored->data,
	//		((double*)(restored->data))[0]);

	// FITS files output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(m31image, "results/m31image.fits");
	status = export_image_to_fits_c(dirty, "results/dirty.fits");
	status = export_image_to_fits_c(psf, "results/psf.fits");
	if(status) printf("WARNING: FITSIO status: %d\n", status);

	return 0;
}
