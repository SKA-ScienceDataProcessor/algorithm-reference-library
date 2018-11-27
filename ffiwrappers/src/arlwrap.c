// Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
//
// Define symbols for the declared wrapper functions -- the definition
// simply fetches the pointer from Python and calls that.
//

#include <Python.h>

#include "../include/arlwrap.h"
#include "../include/wrap_support.h"
#include "../include/wrappingcore.h"

static PyThreadState *_master_state;

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

int arl_handle_error()
{
  int error = 0;
  error = (int)(* ( int (*)())(bk_getfn("arl_handle_error")))();
  return error;
}

void arlvis_vis2proto(const ARLVis *visin, void *visout)
{
  ARLVisPB vispb = ARLVIS_PB__INIT;//(ARLVisPB *)malloc(sizeof(ARLVisPB));
  Data data = DATA__INIT;
//  arlvis_pb__init(&vispb);
  vispb.nvis = visin->nvis;
  vispb.npol = visin->npol;
  data.uvw = ((int *)visin->data)[0]; // uvw
  printf("uvw %d\n", data.uvw);
  data.time = ((int *)visin->data)[1]; // time
  data.frequency = ((int *)visin->data)[2];  // frequency
  data.channel_bandwidth = ((int *)visin->data)[3];  //  channel_bandwidth
  data.integration_time = ((int *)visin->data)[4];
  data.antenna1 = ((int *)visin->data)[5];
  data.antenna2 = ((int *)visin->data)[6];
  data.vis = ((int *)visin->data)[7];
  data.weight = ((int *)visin->data)[8];
  vispb.data = &data;
  vispb.phasecentre = visin->phasecentre;
  visout = malloc(arlvis_pb__get_packed_size(&vispb));
  arlvis_pb__pack(&vispb, visout);
//  BKFNPY(arlvis_vis2proto)(visin, visout);
}

void arlvis_proto2vis(const void *visin, ARLVis *visout)
{
  BKFNPY(arlvis_proto)(visin, visout);
}

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

void arl_invert_2d_proto(const void *visin, const Image *img_in, bool dopsf, Image *out, double *sumwt)
{
  BKFNPY(arl_invert_2d_proto)(visin, img_in, dopsf, out, sumwt);
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

void arl_predict_2d_proto(const ARLVis *visin, const Image *img, void *visout) {
  BKFNPY(arl_predict_2d_proto)(visin, img, visout);
}

void arl_create_image_from_visibility(const ARLVis *vis, Image *model) {
  BKFNPY(arl_create_image_from_visibility)(vis, model);
}
void arl_create_image_from_visibility_proto(const uint8_t *vis, Image *model) {
  BKFNPY(arl_create_image_from_visibility_proto)((uint8_t)vis, model);
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

void arl_invert_function_blockvis(ARLConf * lowconf, const ARLVis *blockvisin, Image * img_model, int vis_slices, Image * img_dirty){
 	BKFNPY(arl_invert_function_blockvis)(lowconf, blockvisin, img_model, vis_slices, img_dirty);
}

