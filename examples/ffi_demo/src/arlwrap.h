// Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
// Header for C-Wrapped version of ARL
//
#ifndef __ARLWRAP_H__
#define __ARLWRAP_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t nvis;
  // Shape: (3, nvis)
  double *uvw;
  double *time;
  double *freq;
  double *bw;
  double *intgt;
  int *a1;
  int *a2;
  float complex *cwis;
  float *wght;
  float *imgwght;
} ARLVis;

#ifdef __cplusplus
}
#endif

#endif
