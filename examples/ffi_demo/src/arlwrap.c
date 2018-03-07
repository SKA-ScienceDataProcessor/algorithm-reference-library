// Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
//
// Define symbols for the declared wrapper functions -- the definition
// simply fetches the pointer from Python and calls that.
//

#include <Python.h>

#include "../include/arlwrap.h"
#include "../include/wrap_support.h"

#ifdef __GNUC__
#define ARL_TLS __thread
#else
#define ARL_TLS
#endif
static ARL_TLS int arl_bypass_ = 0;
static ARL_TLS int arl_bypass_check_ = 0;

static PyThreadState *_master_state;

void void_routine() {
}
// in: fname: same as the *NamedTuple* containing the FFI memory address of a
// Python routine.
// E.G.: Python routine 'arl_restore_cube_fii', NamedTuple 'arl_restore_cube' =>
// => fname == "arl_restore_cube".
//
// This routine imports the arlwrap.py module, loads the NamedTuple named
// 'fname', and extracts the function address. The address is then returned as
// a callable C function pointer.
size_t bk_getfn(const char* fname)
{
  size_t res=0;
  PyObject *m, *pyfn, *fnaddress;

  /* Return immediately if the environment variable
   * has already been checked. */
  if (!arl_bypass_check_) {
      char* flag = getenv("ARL_BYPASS_FFI");
      arl_bypass_check_ = 1;
      if (flag && (
              !strcmp(flag, "TRUE") ||
              !strcmp(flag, "true") ||
              !strcmp(flag, "1") ))
          arl_bypass_ = 1;
  }
	// call 'void' routine if ARL_BYPASS_FFI is set
	if (arl_bypass_) {
		return (size_t)&void_routine;
	}

  PyGILState_STATE gilstate = PyGILState_Ensure();

  if(!(  m= PyImport_ImportModule("arlwrap") ))
    goto failed;
  if(!( pyfn = PyObject_GetAttrString(m, fname)))
    goto failed;
  if(!( fnaddress = PyObject_GetAttrString(pyfn, "address")))
    goto failed;

  res=PyNumber_AsSsize_t(fnaddress, NULL);

  PyGILState_Release(gilstate);
  return res;

 failed:
  PyErr_Print();
  PyGILState_Release(gilstate);  
  return 0;

  // Not decrementing references here since unknown when the objects
  // will be used for C layer. Therefore breakin python functions will
  // never get garbage collected. 
}

void arl_initialize(void)
{
  Py_Initialize();
  PyEval_InitThreads();

	_master_state = PyEval_SaveThread();
}

void arl_finalize(void)
{
	PyEval_RestoreThread(_master_state);
	Py_Finalize();
}


// Not a prototype, so warning will be generated. TODO: switch off
// warning for this instance only?
#define BKFNPY(F)  (* ( void (*)() )(bk_getfn( #F ))) 


void helper_get_image_shape(const double *frequency, double cellsize,
		int *shape)
{
  BKFNPY(helper_get_image_shape)(frequency, cellsize, shape);
}

void helper_get_image_shape_multifreq(ARLConf *lowconf, double cellsize,
		int npixel, int *shape)
{
        BKFNPY(helper_get_image_shape_multifreq)(lowconf, cellsize, npixel, shape);
}

void helper_set_image_params(const ARLVis *vis, Image *image) {
  BKFNPY(helper_set_image_params)(vis, image);
}

void arl_invert_2d(const ARLVis *visin, const Image *img_in, bool dopsf, Image *out, double *sumwt)
{
  BKFNPY(arl_invert_2d)(visin, img_in, dopsf, out, sumwt);
}

void arl_create_visibility(ARLConf *lowconf, ARLVis *res_vis)
{
  BKFNPY(arl_create_visibility)(lowconf, res_vis);
}

void arl_create_blockvisibility(ARLConf *lowconf, ARLVis *res_vis)
{
	BKFNPY(arl_create_blockvisibility)(lowconf, res_vis);
}

void arl_advise_wide_field(ARLConf *lowconf, ARLVis *res_vis, ARLadvice *adv)
{
	BKFNPY(arl_advise_wide_field)(lowconf, res_vis, adv);
}

void arl_create_test_image(const double *frequency, double cellsize, char *phasecentre,
		Image *res_img)
{
  BKFNPY(arl_create_test_image)(frequency, cellsize, phasecentre, res_img);
}

void arl_create_low_test_image_from_gleam(ARLConf *lowconf, double cellsize, int npixel, char *phasecentre,
		Image *res_img)
{
	BKFNPY(arl_create_low_test_image_from_gleam)(lowconf, cellsize, npixel, phasecentre, res_img);
}

/*
void arl_copy_visibility(const ARLVis *visin, ARLVis *visout, bool zero)
{
  BKFNPY(arl_copy_visibility)(visin, visout, zero);
}
*/

void arl_copy_visibility(ARLConf *lowconf, const ARLVis *visin, ARLVis *visout, bool zero)
{
  BKFNPY(arl_copy_visibility)(lowconf, visin, visout, zero);
}

void arl_copy_blockvisibility(ARLConf *lowconf, const ARLVis *visin, ARLVis *visout, int zero)
{
  BKFNPY(arl_copy_blockvisibility)(lowconf, visin, visout, zero);
}

void arl_predict_2d(const ARLVis *visin, const Image *img, ARLVis *visout) {
  BKFNPY(arl_predict_2d)(visin, img, visout);
}

void arl_create_image_from_visibility(const ARLVis *vis, Image *model) {
  BKFNPY(arl_create_image_from_visibility)(vis, model);
}

void arl_create_image_from_blockvisibility(ARLConf *lowconf, const ARLVis *blockvis, double cellsize, int npixel, char* phasecentre, Image *model){
	BKFNPY(arl_create_image_from_blockvisibility)(lowconf, blockvis, cellsize, npixel, phasecentre, model);
}

void arl_deconvolve_cube(Image *dirty, Image *psf, Image *restored, Image *residual)
{
  BKFNPY(arl_deconvolve_cube)(dirty, psf, restored, residual);
}

void arl_deconvolve_cube_ical(Image *dirty, Image *psf, Image *restored, Image *residual)
{
  BKFNPY(arl_deconvolve_cube_ical)(dirty, psf, restored, residual);
}

void arl_restore_cube(Image *model, Image *psf, Image *residual, Image *restored)
{
  BKFNPY(arl_restore_cube)(model, psf, residual, restored);
}

void arl_restore_cube_ical(Image *model, Image *psf, Image *residual, Image *restored)
{
  BKFNPY(arl_restore_cube_ical)(model, psf, residual, restored);
}

void helper_get_nbases(char * config_name, ant_t * nbases)
{
  BKFNPY(helper_get_nbases)(config_name, nbases);
}

void helper_get_nbases_rmax(char * config_name, double rmax, ant_t * nbases)
{
  BKFNPY(helper_get_nbases_rmax)(config_name, rmax, nbases);
}

void arl_predict_function(ARLConf *lowconf, const ARLVis *visin, const Image *img, ARLVis *visout, ARLVis *blockvisout, long long int *cindexout) {
	BKFNPY(arl_predict_function)(lowconf, visin, img, visout, blockvisout, cindexout);
}

void arl_predict_function_ical(ARLConf *lowconf, ARLVis *visinout, const Image *img, ARLVis *blockvisinout, long long int *cindexinout, int vis_slices) {
	BKFNPY(arl_predict_function_ical)(lowconf, visinout, img, blockvisinout, cindexinout, vis_slices);
}

void arl_invert_function(ARLConf * lowconf, const ARLVis *visin, Image * img_model, int vis_slices, Image * img_dirty){
	BKFNPY(arl_invert_function)(lowconf, visin, img_model, vis_slices, img_dirty);
}

void arl_invert_function_ical(ARLConf * lowconf, const ARLVis *visin, Image * img_model, int vis_slices, Image * img_dirty){
	BKFNPY(arl_invert_function_ical)(lowconf, visin, img_model, vis_slices, img_dirty);
}

void arl_invert_function_psf(ARLConf * lowconf, const ARLVis *visin, Image * img_model, int vis_slices, Image * img_dirty){
	BKFNPY(arl_invert_function_psf)(lowconf, visin, img_model, vis_slices, img_dirty);
}

void arl_ical(ARLConf * lowconf, const ARLVis *visin, Image * img_model, int vis_slices, Image * img_deconv, Image * img_resid, Image * img_rest){
	BKFNPY(arl_ical)(lowconf, visin, img_model, vis_slices, img_deconv, img_resid, img_rest);
}

void arl_convert_visibility_to_blockvisibility(ARLConf *lowconf, const ARLVis *visin, const ARLVis *blockvisin, long long int *cindexin, ARLVis *visout) {
	BKFNPY(arl_convert_visibility_to_blockvisibility)(lowconf, visin, blockvisin, cindexin, visout);
}

void arl_convert_blockvisibility_to_visibility(ARLConf *lowconf, const ARLVis *blockvisin, ARLVis *visout, long long int *cindexout, ARLVis *blockvisout) {
	BKFNPY(arl_convert_blockvisibility_to_visibility)(lowconf, blockvisin, visout, cindexout, blockvisout);
}

void arl_create_gaintable_from_blockvisibility(ARLConf *lowconf, const ARLVis *visin, ARLGt *gtout) {
	BKFNPY(arl_create_gaintable_from_blockvisibility)(lowconf, visin, gtout);
}

void arl_apply_gaintable(ARLConf *lowconf, const ARLVis *visin, ARLGt *gtin, ARLVis *visout, int inverse) {
	BKFNPY(arl_apply_gaintable)(lowconf, visin, gtin, visout, inverse);
}

void arl_apply_gaintable_ical(ARLConf *lowconf, ARLVis *visin, ARLGt *gtin, int inverse) {
	BKFNPY(arl_apply_gaintable_ical)(lowconf, visin, gtin, inverse);
}

void arl_simulate_gaintable(ARLConf *lowconf, ARLGt *gt) {
	BKFNPY(arl_simulate_gaintable)(lowconf, gt);
}

void arl_solve_gaintable_ical(ARLConf *lowconf, const ARLVis *blockvisin, const ARLVis *blockvisin_pred, ARLGt *gt, int vis_slices){
	BKFNPY(arl_solve_gaintable_ical)(lowconf, blockvisin, blockvisin_pred, gt, vis_slices);
}

void arl_set_visibility_data_to_zero(ARLConf * lowconf, ARLVis * vis) {
	BKFNPY(arl_set_visibility_data_to_zero)(lowconf, vis);
}

void arl_manipulate_visibility_data(ARLConf *lowconf, const ARLVis *vis1in, const ARLVis *vis2in, ARLVis *visout, int op) {
	BKFNPY(arl_manipulate_visibility_data)(lowconf, vis1in, vis2in, visout, op);
}

void arl_add_to_model(Image* model, Image* res) {
	BKFNPY(arl_add_to_model)(model, res);
}

void arl_predict_function_blockvis(ARLConf * lowconf, ARLVis * visin, const Image * img) {
	BKFNPY(arl_predict_function_blockvis)(lowconf, visin, img);
}


