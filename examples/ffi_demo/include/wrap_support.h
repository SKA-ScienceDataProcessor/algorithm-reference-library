#ifndef WRAP_SUPPORT_H
#define WRAP_SUPPORT_H

#include "arlwrap.h"

int export_image_to_fits_c(Image *im, char * filename);
Image *allocate_image(int *shape);
Image *destroy_image(Image *image);
ARLConf *allocate_arlconf_default(const char *conf_name);
ARLVis *allocate_vis_data(int npol, int nvis);
ARLVis *allocate_blockvis_data(int nants, int nchan, int npol, int ntimes);
ARLGt *allocate_gt_data(int nant, int nchan, int nrec, int ntimes);
ARLVis *destroy_vis(ARLVis *vis);
ARLGt *destroy_gt(ARLGt *gt);

#endif
