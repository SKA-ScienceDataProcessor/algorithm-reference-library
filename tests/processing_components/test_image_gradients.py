"""Unit tests for testing support

"""

import logging
import unittest

from data_models.parameters import arl_path
from processing_components.image.gradients import image_gradients
from processing_components.image.operations import export_image_to_fits, show_image, import_image_from_fits

log = logging.getLogger(__name__)


class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def test_create_gradient(self):
        real_vp = import_image_from_fits(arl_path('data/models/MID_GRASP_VP_real.fits'))
        gradx, grady = image_gradients(real_vp)
        
        gradxx, gradxy = image_gradients(gradx)
        gradyx, gradyy = image_gradients(grady)

        gradx.data *= real_vp.data
        grady.data *= real_vp.data
        gradxx.data *= real_vp.data
        gradxy.data *= real_vp.data
        gradyx.data *= real_vp.data
        gradyy.data *= real_vp.data

        import matplotlib.pyplot as plt
        plt.clf()
        show_image(gradx, title='gradx')
        plt.show()
        plt.clf()
        show_image(grady, title='grady')
        plt.show()
        export_image_to_fits(gradx, "%s/test_image_gradients_gradx.fits" % (self.dir))
        export_image_to_fits(grady, "%s/test_image_gradients_grady.fits" % (self.dir))

        plt.clf()
        show_image(gradxx, title='gradxx')
        plt.show()
        plt.clf()
        show_image(gradxy, title='gradxy')
        plt.show()
        plt.clf()
        show_image(gradyx, title='gradyx')
        plt.show()
        plt.clf()
        show_image(gradyy, title='gradyy')
        plt.show()
        export_image_to_fits(gradxx, "%s/test_image_gradients_gradxx.fits" % (self.dir))
        export_image_to_fits(gradxy, "%s/test_image_gradients_gradxy.fits" % (self.dir))
        export_image_to_fits(gradyx, "%s/test_image_gradients_gradyx.fits" % (self.dir))
        export_image_to_fits(gradyy, "%s/test_image_gradients_gradyy.fits" % (self.dir))


if __name__ == '__main__':
    unittest.main()
