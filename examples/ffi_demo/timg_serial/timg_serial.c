#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "../include/arlwrap.h"
#include "../include/wrap_support.h"

/*
 * Verifies that:
 * - vt and vtmp are unique in memory
 * - vt and vtmp have equivalent values
 */
int verify_arl_copy(ARLVis *vt, ARLVis *vtmp)
{
	char *vtdata_bytes, *vtmpdata_bytes;
	int ARLVisDataSize;
	int i;

	if (vt == vtmp) {
		fprintf(stderr, "vt == vtmp\n");
		return 1;
	}

	if (!((vt->nvis == vtmp->nvis) && (vt->npol == vtmp->npol))) {
		return 2;
	}

	if (vt->data == vtmp->data) {
		return 3;
	}

	ARLVisDataSize = 80 + (32 * vt->npol * vt->nvis);
	vtdata_bytes = (char*) vt->data;
	vtmpdata_bytes = (char*) vtmp->data;

	for (i=0; i<ARLVisDataSize; i++) {
		if (vtdata_bytes[i] != vtmpdata_bytes[i]) {
			return 4;
		}
	}

	return 0;
}

int main(int argc, char **argv)
{
	int *shape = malloc(4*sizeof(int));
	int status;
	int nvis;

	double cellsize = 0.0005;
	char config_name[] = "LOWBD2-CORE";

	ARLVis *vt;
	ARLVis *vtmodel;
	ARLVis *vtmp;

	ARLConf *lowconfig;

	arl_initialize();

	lowconfig = allocate_arlconf_default(config_name);

	nvis = (lowconfig->nbases)*(lowconfig->nfreqs)*(lowconfig->ntimes);

	vt = allocate_vis_data(lowconfig->npol, nvis);
	vtmp = allocate_vis_data(lowconfig->npol, nvis);

	// Calculate shape of future images, store in 'shape'
	helper_get_image_shape(lowconfig->freqs, cellsize, shape);

	Image *model = allocate_image(shape);
	Image *m31image = allocate_image(shape);
	Image *dirty = allocate_image(shape);
	Image *psf = allocate_image(shape);
	Image *comp = allocate_image(shape);
	Image *residual = allocate_image(shape);
	Image *restored = allocate_image(shape);

	arl_create_visibility(lowconfig, vt);

	/* TODO the vt->phasecentre part should be moved to a separate routine */
	arl_create_test_image(lowconfig->freqs, cellsize, vt->phasecentre, m31image);

	arl_predict_2d(vt, m31image, vtmp);

	vt = destroy_vis(vt);
	vt = vtmp;
	vtmp = NULL;


	arl_create_image_from_visibility(vt, model);

	double *sumwt = malloc(sizeof(double));

	arl_invert_2d(vt, model, false, dirty, sumwt);
	arl_invert_2d(vt, model, true, psf, sumwt);

	arl_deconvolve_cube(dirty, psf, comp, residual);
	arl_restore_cube(comp, psf, residual, restored);

	// FITS files output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(m31image, "results/m31image.fits");
	status = export_image_to_fits_c(dirty, "results/dirty.fits");
	status = export_image_to_fits_c(psf, "results/psf.fits");
	status = export_image_to_fits_c(residual, "results/residual.fits");
	status = export_image_to_fits_c(restored, "results/restored.fits");
	status = export_image_to_fits_c(comp, "results/solution.fits");

	model = destroy_image(model);
	m31image = destroy_image(m31image);
	dirty = destroy_image(dirty);
	psf = destroy_image(psf);
	residual = destroy_image(residual);
	restored = destroy_image(restored);

	vtmodel = allocate_vis_data(lowconfig->npol, nvis);
	vtmp = allocate_vis_data(lowconfig->npol, nvis);

	arl_create_visibility(lowconfig, vtmodel);

	arl_predict_2d(vtmodel, comp, vtmp);

	vtmodel = destroy_vis(vtmodel);
	vtmp = destroy_vis(vtmp);

	comp = destroy_image(comp);

	arl_finalize();

	return 0;
}
