#include "ical_pu_routines.h"

/* Example kernel codelet: calls create_visibility, specifies number of buffers
 * (=arguments), and set argument read/write modes
 */
struct starpu_codelet create_blockvisibility_cl = {
	.cpu_funcs = { pu_create_blockvisibility },
	.name = "pu_create_blockvisibility",
	.nbuffers = 2,
	.modes = { STARPU_RW, STARPU_W }
};

struct starpu_codelet create_visibility_cl = {
	.cpu_funcs = { pu_create_visibility },
	.name = "pu_create_visibility",
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_W }
};

struct starpu_codelet create_test_image_cl = {
	.cpu_funcs = { pu_create_test_image },
	.name = "pu_create_test_image",
	.nbuffers = 4,
	.modes = { STARPU_R, STARPU_R, STARPU_W, STARPU_W }
};

struct starpu_codelet predict_2d_cl = {
	.cpu_funcs = { pu_predict_2d },
	.name = "pu_predict_2d",
	.nbuffers = 3,
	.modes = { STARPU_R, STARPU_R, STARPU_W }
};

struct starpu_codelet create_from_visibility_cl = {
	.cpu_funcs = { pu_create_from_visibility },
	.name = "pu_create_from_visibility",
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_W }
};

struct starpu_codelet invert_2d_cl = {
	.cpu_funcs = { pu_invert_2d },
	.name = "pu_invert_2d",
	.nbuffers = 5,
	.modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_RW }
};

struct starpu_codelet deconvolve_cube_cl = {
	.cpu_funcs = { pu_deconvolve_cube },
	.name = "pu_deconvolve_cube",
	.nbuffers = 4,
	.modes = { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW }
};

struct starpu_codelet restore_cube_cl = {
	.cpu_funcs = { pu_restore_cube },
	.name = "pu_restore_cube",
	.nbuffers = 4,
	.modes = { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW }
};

struct starpu_codelet advise_wide_field_cl = {
  .cpu_funcs = { pu_advise_wide_field },
  .name = "pu_advise_wide_field",
  .nbuffers = 3,
  .modes = { STARPU_RW, STARPU_R, STARPU_W}
};

struct starpu_codelet helper_get_image_shape_multifreq_cl = {
  .cpu_funcs = { pu_helper_get_image_shape_multifreq },
  .name = "pu_helper_get_image_shape_multifreq",
  .nbuffers = 4,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_W }
};

struct starpu_codelet allocate_image_cl = {
  .cpu_funcs = { pu_allocate_image },
  .name = "pu_allocate_image",
  .nbuffers = 2,
  .modes = { STARPU_R, STARPU_RW}
};

struct starpu_codelet create_low_test_image_from_gleam_cl = {
  .cpu_funcs = { pu_create_low_test_image_from_gleam },
  .name = "pu_create_low_test_image_from_gleam",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW }
};

struct starpu_codelet predict_function_blockvis_cl = {
  .cpu_funcs = { pu_predict_function_blockvis },
  .name = "pu_predict_function_blockvis",
  .nbuffers = 3,
  .modes = { STARPU_R, STARPU_RW, STARPU_RW }
};

struct starpu_codelet create_gaintable_from_blockvisibility_cl = {
  .cpu_funcs = { pu_create_gaintable_from_blockvisibility },
  .name = "pu_create_gaintable_from_blockvisibility",
  .nbuffers = 3,
  .modes = { STARPU_R,STARPU_R, STARPU_W }
};

struct starpu_codelet simulate_gaintable_cl = {
  .cpu_funcs = { pu_simulate_gaintable },
  .name = "pu_simulate_gaintable",
  .nbuffers = 2,
  .modes = { STARPU_R, STARPU_RW }
};

struct starpu_codelet apply_gaintable_cl = {
  .cpu_funcs = { pu_apply_gaintable },
  .name = "pu_apply_gaintable",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW, STARPU_R }
};

struct starpu_codelet create_image_from_blockvisibility_cl = {
  .cpu_funcs = { pu_create_image_from_blockvisibility },
  .name = "pu_create_image_from_blockvisibility",
  .nbuffers = 6,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW, STARPU_RW }
};

struct starpu_codelet invert_function_blockvis_cl = {
  .cpu_funcs = { pu_invert_function_blockvis },
  .name = "pu_invert_function_blockvis",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW }
};

struct starpu_codelet convert_blockvisibility_to_visibility_cl = {
  .cpu_funcs = { pu_convert_blockvisibility_to_visibility },
  .name = "pu_convert_blockvisibility_to_visibility",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW, STARPU_RW }
};

struct starpu_codelet copy_blockvisibility_cl = {
  .cpu_funcs = { pu_copy_blockvisibility },
  .name = "pu_copy_blockvisibility",
  .nbuffers = 4,
  .modes = { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW }
};

struct starpu_codelet add_to_model_cl = {
  .cpu_funcs = { pu_add_to_model },
  .name = "pu_add_to_model",
  .nbuffers = 2,
  .modes = { STARPU_R, STARPU_RW }
};

struct starpu_codelet set_visibility_data_to_zero_cl = {
  .cpu_funcs = { pu_set_visibility_data_to_zero },
  .name = "pu_set_visibility_data_to_zero",
  .nbuffers = 2,
  .modes = { STARPU_R, STARPU_RW }
};

struct starpu_codelet predict_function_ical_cl = {
  .cpu_funcs = { pu_predict_function_ical },
  .name = "pu_predict_function_ical",
  .nbuffers = 6,
  .modes = { STARPU_R, STARPU_RW, STARPU_R, STARPU_RW, STARPU_RW, STARPU_RW }
};

struct starpu_codelet convert_visibility_to_blockvisibility_cl = {
  .cpu_funcs = { pu_convert_visibility_to_blockvisibility },
  .name = "pu_convert_visibility_to_blockvisibility",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW }
};

struct starpu_codelet manipulate_visibility_data_cl = {
  .cpu_funcs = { pu_manipulate_visibility_data },
  .name = "pu_manipulate_visibility_data",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_RW, STARPU_RW }
};

struct starpu_codelet invert_function_ical_cl = {
  .cpu_funcs = { pu_invert_function_ical },
  .name = "pu_invert_function_ical",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW }
};

struct starpu_codelet invert_function_psf_cl = {
  .cpu_funcs = { pu_invert_function_psf },
  .name = "pu_invert_function_psf",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW }
};

struct starpu_codelet deconvolve_cube_ical_cl = {
  .cpu_funcs = { pu_deconvolve_cube_ical },
  .name = "pu_deconvolve_cube_ical",
  .nbuffers = 4,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_RW }
};
/*
struct starpu_codelet set_visibility_data_to_zero_cl = {
  .cpu_funcs = { pu_set_visibility_data_to_zero },
  .name = "pu_set_visibility_data_to_zero",
  .nbuffers = 2,
  .modes = { STARPU_R, STARPU_RW }
};

struct starpu_codelet _cl = {
  .cpu_funcs = { pu_ },
  .name = "pu_",
  .nbuffers = ,
  .modes = { STARPU_R, STARPU_RW }
};

struct starpu_codelet _cl = {
  .cpu_funcs = { pu_ },
  .name = "pu_",
  .nbuffers = ,
  .modes = { STARPU_R, STARPU_RW }
};

struct starpu_codelet _cl = {
  .cpu_funcs = { pu_ },
  .name = "pu_",
  .nbuffers = ,
  .modes = { STARPU_R, STARPU_RW }
};
*/
