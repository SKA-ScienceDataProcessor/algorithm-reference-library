// Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
// Header for C-Wrapped version of ARL
//
#ifndef __ARLWRAP_H__
#define __ARLWRAP_H__

#include <complex.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t nvis;
  int npol;
  // This needs to be interpret differently dependent on the value of
  // the npol. For example when npol is 4, data is equivalent to "C"
  // type type "ARLVisEntryP4[nvis]"
  void *data;
	char *pickle;
} ARLVis;

// This is what the data array for four polarisations look like. Can
// recast data to this (or equivalent of 2 or 1 polarisation if really
// needed. This memory layout show allow re-use in numpy without any
// data copying.
typedef struct {
  double uvw[3];
  double time;
  double freq;
  double bw;
  double intgt;
  int a1;
  int a2;
  float complex vis[4];
  float wght[4];
  float imgwght [4];
} ARLVisEntryP4;

  void arl_copy_visibility(const ARLVis *visin,
			   ARLVis *visout,
			   bool zero);

typedef struct {
	size_t size;
	int data_shape[4];
	void *data;
	char *wcs;
	char *polarisation_frame;
} Image;

void helper_get_image_shape(const double *frequency, double cellsize,
		int *shape);

void arl_create_visibility(const char *lowcore_name, double *times, double *frequency, double *channel_bandwidth, ARLVis *res_vis);
void arl_create_test_image(const double *frequency, double cellsize,
		Image *res_img);
void arl_predict_2d(const ARLVis *visin, const Image *img, ARLVis *visout);
void arl_invert_2d(const ARLVis *visin, const Image *img_in, bool dopsf, Image *out, double *sumwt);
void arl_create_image_from_visibility(const ARLVis *vis, Image *model);
void arl_deconvolve_cube(Image *dirty, Image *psf, Image *restored,
		Image *residual);
void arl_restore_cube(Image *model, Image *psf, Image *residual,
		Image *restored);

#ifdef __cplusplus
}
#endif

#endif
