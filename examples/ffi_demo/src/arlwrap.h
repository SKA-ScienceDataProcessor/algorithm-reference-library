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
  int npol;
  // This needs to be interpret differently dependent on the value of
  // the npol. For example when npol is 4, data is equivalent to "C"
  // type type "ARLVisEntryP4[nvis]"
  void *data;
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

#ifdef __cplusplus
}
#endif

#endif
