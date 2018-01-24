#include <stdlib.h>
#include <fitsio.h>

#include "arlwrap.h"


/* Export image to FITS */
/* Assuming nx*ny*nfreq */
/* ToDo - add polarization and wcs */
int export_image_to_fits_c(Image *im, char * filename)
{
	int status = 0, exists;
	fitsfile *fptr;       /* pointer to the FITS file; defined in fitsio.h */
	long  fpixel = 1, naxis = 4, nelements;
	long naxes[4];

	naxes[0] = im->data_shape[3];
	naxes[1] = im->data_shape[2];
	naxes[2] = im->data_shape[1];
	naxes[3] = im->data_shape[0];

	fits_file_exists(filename, &exists, &status); /* check if the file exists */

	if(exists != 0) {
		fits_open_file(&fptr, filename, READWRITE, &status); /* open existed file */
	}
	else {
		fits_create_file(&fptr, filename, &status);   /* create new file */
	}

	/* Create the primary array image  */
	fits_create_img(fptr, DOUBLE_IMG, naxis, naxes, &status);
	nelements = naxes[0] * naxes[1] * naxes[2] * naxes[3];          /* number of pixels to write */
	/* Write the array of integers to the image */
	fits_write_img(fptr, TDOUBLE, fpixel, nelements, im->data, &status);
	fits_close_file(fptr, &status);            /* close the file */
	fits_report_error(stderr, status);  /* print out any error messages */
	return status;
}

// Allocate memory for a FITS image structure in 4 dimensions.
// wcs and polarisation_frame store pickled versions of the corresponding Python
// data, currently the sizes are 'magic numbers' found through experimentation -
// we haven't found a way to properly determine the required sizes yet.
Image *allocate_image(int *shape)
{
	int i;
	Image *image;
	
	if (!(image = malloc(sizeof(Image)))) {
		return NULL;
	}

	image->size = 1;

	for(i=0; i<4; i++) {
		image->data_shape[i] = shape[i];
		image->size *= shape[i];
	}

	if(!(image->data = calloc(image->size,sizeof(double)))) {
		free(image);
		return NULL;
	}
	if(!(image->wcs = calloc(2997,sizeof(char)))) {
		free(image->data);
		free(image);
		return NULL;
	}
	if(!(image->polarisation_frame = calloc(115,sizeof(char)))) {
		free(image->data);
		free(image->wcs);
		free(image);
		return NULL;
	}

	return image;
}

void *destroy_image(Image *image)
{
	if (image) {
		if(image->data) {
			free(image->data);
		}
		if(image->wcs) {
			free(image->wcs);
		}
		if(image->polarisation_frame) {
			free(image->polarisation_frame);
		}

		free(image);
	}

	return NULL;
}

ARLConf *allocate_arlconf_default(const char *conf_name)
{
	ARLConf *config;
	ant_t nb;

	if (!(config = malloc(sizeof(ARLConf)))) {
		return NULL;
	}

	// Find out the number of the antennas and the baselines, keep in nb structure
	//nb.nbases = 1;
	helper_get_nbases(conf_name, &nb);

	// Assigning configuraion values
	config->confname = conf_name;
	config->pc_ra = 15.0;
	config->pc_dec = -45.0;
	config->times = calloc(1, sizeof(double));
	config->ntimes = 1;
	config->freqs = malloc(sizeof(double));	
	config->nfreqs = 1;	
	config->channel_bandwidth = malloc(sizeof(double));	
	config->nchanwidth = 1;
	config->nbases = nb.nbases;
	config->npol = 1;

	config->freqs[0] = 1e8;
	config->channel_bandwidth[0] = 1e6;

	return config;
}

ARLVis *allocate_vis_data(int npol, int nvis)
{
	ARLVis *vis;
	if (!(vis = malloc(sizeof(ARLVis)))) {
		return NULL;
	}

	vis->nvis = nvis;
	vis->npol = npol;

	// (80 bytes static data + 32 bytes * npol) * nvis
	if (!(vis->data = malloc((80+(32*npol))*nvis * sizeof(char)))) {
		free(vis);
		return NULL;
	}
	// pickled phasecentre. Size found through experimentation
	if (!(vis->phasecentre = malloc(5000*sizeof(char)))) {
		free(vis->data);
		free(vis);
		return NULL;
	}

	return vis;
}

ARLVis *destroy_vis(ARLVis *vis)
{
	free(vis->phasecentre);
	free(vis->data);
	free(vis);

	return NULL;
}
