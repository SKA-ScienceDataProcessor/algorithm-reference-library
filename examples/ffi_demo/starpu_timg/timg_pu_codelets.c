#include "timg_pu_routines.h"

/* Example kernel codelet: calls create_visibility, specifies number of buffers
 * (=arguments), and set argument read/write modes
 */
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
