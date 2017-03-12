import unittest

from numpy import random
from arl.data.polarisation import *
from numpy.testing import *


class TestPolarisation(unittest.TestCase):
    def test_polarisation_frame(self):
        for frame in ['circular', 'circularnp', 'linear', 'linearnp', 'stokesIQUV', 'stokesIV', 'stokesIQ', 'stokesI']:
            polarisation_frame = Polarisation_Frame(frame)
            assert polarisation_frame.type == frame
            
        assert Polarisation_Frame("circular").npol == 4
        assert Polarisation_Frame("circularnp").npol == 2
        assert Polarisation_Frame("linear").npol == 4
        assert Polarisation_Frame("linearnp").npol == 2
        assert Polarisation_Frame("circular").npol == 4
        assert Polarisation_Frame("stokesI").npol == 1

        with self.assertRaises(RuntimeError):
            polarisation_frame = Polarisation_Frame("circuloid")
    
    def test_rec_frame(self):
        rec_frame = Receptor_Frame("linear")
        assert rec_frame.nrec == 2
        
        rec_frame = Receptor_Frame("circular")
        assert rec_frame.nrec == 2

        rec_frame = Receptor_Frame("stokesI")
        assert rec_frame.nrec == 1

        with self.assertRaises(RuntimeError):
            rec_frame = Receptor_Frame("circuloid")
    
    def test_correlate(self):
        
        for frame in ["linear", "circular", "stokesI"]:
            rec_frame = Receptor_Frame(frame)
            assert correlate_polarisation(rec_frame) == Polarisation_Frame(frame)

    def test_congruent(self):
    
        for frame in ["linear", "circular", "stokesI"]:
            assert congruent_polarisation(Receptor_Frame(frame), Polarisation_Frame(frame))
            assert not congruent_polarisation(Receptor_Frame(frame), Polarisation_Frame("stokesIQUV"))

    def test_stokes_linear_conversion(self):
    
        stokes = numpy.array([1.0, 0.0, 0.0, 0.0])
        linear = convert_stokes_to_linear(stokes)
        assert_array_almost_equal(linear, numpy.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j]))
    
        stokes = numpy.array([0.0, 1.0, 0.0, 0.0])
        linear = convert_stokes_to_linear(stokes)
        assert_array_almost_equal(linear, numpy.array([1.0 + 0j, 0j, 0j, -1.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 1.0, 0.0])
        linear = convert_stokes_to_linear(stokes)
        assert_array_almost_equal(linear, numpy.array([0.0 + 0j, 1.0 + 0j, 1.0 + 0j, 0.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 0.0, 1.0])
        linear = convert_stokes_to_linear(stokes)
        assert_array_almost_equal(linear, numpy.array([0.0 + 0j, +1.0j, -1.0j, 0.0 + 0j]))

    def test_stokes_circular_conversion(self):
    
        stokes = numpy.array([1.0, 0.0, 0.0, 0.0])
        circular = convert_stokes_to_circular(stokes)
        assert_array_almost_equal(circular, numpy.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j]))
    
        stokes = numpy.array([0.0, 1.0, 0.0, 0.0])
        circular = convert_stokes_to_circular(stokes)
        assert_array_almost_equal(circular, numpy.array([0.0 + 0j, -1j, -1j, 0.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 1.0, 0.0])
        circular = convert_stokes_to_circular(stokes)
        assert_array_almost_equal(circular, numpy.array([0.0 + 0j, 1.0 + 0j, -1.0 + 0j, 0.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 0.0, 1.0])
        circular = convert_stokes_to_circular(stokes)
        assert_array_almost_equal(circular, numpy.array([1.0 + 0j, +0.0j, 0.0j, -1.0 + 0j]))

    def test_stokes_linear_stokes_conversion(self):
    
        stokes = numpy.array([1, 0.5, 0.2, -0.1])
        linear = convert_stokes_to_linear(stokes)
        assert_array_almost_equal(convert_linear_to_stokes(linear).real, stokes, 15)

    def test_stokes_circular_stokes_conversion(self):
    
        stokes = numpy.array([1, 0.5, 0.2, -0.1])
        circular = convert_stokes_to_circular(stokes)
        assert_array_almost_equal(convert_circular_to_stokes(circular).real, stokes, 15)

    def test_image_conversion(self):
    
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        cir = convert_stokes_to_circular(stokes)
        st = convert_circular_to_stokes(cir)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_image_auto_conversion(self):
    
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        ipf = Polarisation_Frame('stokesIQUV')
        opf = Polarisation_Frame('circular')
        cir = convert_pol_frame(stokes, ipf, opf)
        st = convert_pol_frame(cir, opf, ipf)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_image_auto_conversion_I(self):
    
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        ipf = Polarisation_Frame('stokesI')
        opf = Polarisation_Frame('stokesI')
        cir = convert_pol_frame(stokes, ipf, opf)
        st = convert_pol_frame(cir, opf, ipf)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_vis_conversion(self):

        stokes = numpy.array(random.uniform(-1.0, 1.0, [1000,3,4]))
        cir = convert_stokes_to_circular(stokes, polaxis=2)
        st = convert_circular_to_stokes(cir, polaxis=2)
        assert_array_almost_equal(st.real, stokes, 15)
        

    def test_vis_auto_conversion(self):

        stokes = numpy.array(random.uniform(-1.0, 1.0, [1000,3,4]))
        ipf = Polarisation_Frame('stokesIQUV')
        opf = Polarisation_Frame('circular')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=2)
        st = convert_pol_frame(cir, opf, ipf, polaxis=2)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_vis_auto_conversion_I(self):

        stokes = numpy.array(random.uniform(-1.0, 1.0, [1000,3,1]))
        ipf = Polarisation_Frame('stokesI')
        opf = Polarisation_Frame('stokesI')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=2)
        st = convert_pol_frame(cir, opf, ipf, polaxis=2)
        assert_array_almost_equal(st.real, stokes, 15)



if __name__ == '__main__':
    unittest.main()
