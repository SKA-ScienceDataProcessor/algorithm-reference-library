
from arl.synthesis_support import *
from crocodile.simulate import *

import unittest
import itertools
import numpy as np
from numpy.testing import assert_allclose

class TestSynthesis(unittest.TestCase):

    def test_coordinates(self):
        for N in [4,5,6,7,8,9]:
            low, high = coordinateBounds(N)
            c = coordinates(N)
            cy, cx = coordinates2(N)
            self.assertAlmostEqual(np.min(c), low)
            self.assertAlmostEqual(np.max(c), high)
            self.assertAlmostEqual(np.min(cx), low)
            self.assertAlmostEqual(np.max(cx), high)
            self.assertAlmostEqual(np.min(cy), low)
            self.assertAlmostEqual(np.max(cy), high)
            self.assertAlmostEqual(c[N//2], 0)
            assert (cx[:,N//2] == 0).all()
            assert (cy[N//2,:] == 0).all()

    def _pattern(self, N):
        return coordinates2(N)[0]+coordinates2(N)[1]*1j

    def test_pad_extract(self):
        for N, N2 in [ (1,1), (1,2), (2,3), (3,4), (2,5), (4,6) ]:
            cs = 1 + self._pattern(N)
            cs_pad = pad_mid(cs, N2)
            cs2 = 1 + self._pattern(N2) * N2 / N
            # At this point all fields in cs2 and cs_pad should either
            # be equal or zero.
            equal = numpy.abs(cs_pad - cs2) < 1e-15
            zero = numpy.abs(cs_pad) < 1e-15
            assert numpy.all(equal + zero)
            # And extracting the middle should recover the original data
            assert_allclose(extract_mid(cs_pad, N), cs)

    def test_extract_oversampled(self):
        for N, Qpx in [ (1,2), (2,3), (3,2), (4,2), (5,3) ]:
            a = 1+self._pattern(N * Qpx)
            ex = extract_oversampled(a, 0, 0, Qpx, N)/Qpx**2
            assert_allclose(ex, 1+self._pattern(N))

    def test_anti_aliasing(self):
        for shape in [(4,4),(5,5),(4,6),(7,3)]:
            aaf = anti_aliasing_function(shape, 0, 10)
            self.assertEqual(aaf.shape, shape)
            self.assertAlmostEqual(aaf[shape[0]//2,shape[1]//2], 1)

    def test_w_kernel_function(self):
        assert_allclose(w_kernel_function(5,0.1,0), 1)
        self.assertAlmostEqual(w_kernel_function(5,0.1,100)[2,2], 1)
        self.assertAlmostEqual(w_kernel_function(10,0.1,100)[5,5], 1)
        self.assertAlmostEqual(w_kernel_function(11,0.1,1000)[5,5], 1)

    def test_kernel_oversampled_subgrid(self):
        # Oversampling should produce the same values where sub-grids overlap
        for N in range(3,30):
            pat = self._pattern(N)
            kern = kernel_oversample(pat, N, 1, N-2)
            kern2 = kernel_oversample(pat, N, 2, N-2)
            assert_allclose(kern[0,0], kern2[0,0], atol=1e-15)
            kern3 = kernel_oversample(pat, N, 3, N-2)
            assert_allclose(kern[0,0], kern3[0,0], atol=1e-15)
            kern4 = kernel_oversample(pat, N, 4, N-2)
            for ux, uy in itertools.product(range(2), range(2)):
                assert_allclose(kern2[uy,ux], kern4[2*uy,2*ux], atol=1e-15)
            kern8 = kernel_oversample(pat, N, 8, N-2)
            for ux, uy in itertools.product(range(3), range(3)):
                assert_allclose(kern4[uy,ux], kern8[2*uy,2*ux], atol=1e-15)

    def test_kernel_scale(self):
        # Scaling the grid should not make a difference
        N = 10
        wff = numpy.zeros((N,N))
        wff[N//2,N//2] = 1 # Not the most interesting kernel...
        k = kernel_oversample(wff, N, 1, N)
        k2 = kernel_oversample(wff, N*2, 1, N)
        assert_allclose(k, k2*4)

    def test_w_kernel_normalisation(self):
        # Test w-kernel normalisation. This isn't quite perfect.
        for Qpx in [4,5,6]:
            for N in [3,5,9,16,20,24,32,64]:
                k = kernel_oversample(w_kernel_function(N+2,0.1,N*10), N+2, Qpx, N)
                assert_allclose(numpy.sum(k), Qpx**2,
                                rtol=0.07)

    def _uvw(self, N, uw=0, vw=0):
        u,v = coordinates2(N)
        u=numpy.hstack(u)
        v=numpy.hstack(v)
        w = uw*u + vw*v
        return numpy.transpose([u,v,w])

    def test_grid_degrid(self):
        # The uvw we chose here correspond exactly with grid points
        # (at w=0) This is the "perfect" case in which all of this
        # should gracefully degrade to a simple FFT. There is
        # especially no information loss, so we can test
        # exhaustively.
        for lam in [6, 1e15]:
          gcf = numpy.conj(w_kernel(1/lam, 0, 1, 1, 1))
          for N in range(2,6):
            uvw = self._uvw(N)*lam
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                # Simulate and grid a single point source
                vis = simulate_point(uvw, x/lam, y/lam)
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvw/lam, vis)
                # Do it using convolution gridding too, which should
                # double the result
                convgrid(gcf, a, uvw/lam, vis)
                a /= 2
                # Image should have a single 1 at the source
                img = numpy.real(ifft(a))
                self.assertAlmostEqual(img[N//2+y,N//2+x], 1)
                img[N//2+y,N//2+x] = 0
                assert_allclose(img, 0, atol=1e-14)
                # Degridding should reproduce the original visibilities
                vis_d = degrid(a, uvw/lam)
                assert_allclose(vis, vis_d)
                vis_d = convdegrid(gcf, a, uvw/lam)
                assert_allclose(vis, vis_d)

    def test_grid_degrid_shift(self):
        lam = 100
        for N in range(3,7):
          for dl, dm in [(1/lam, 1/lam), (-1/lam, 2/lam), (5/lam, 0)]:
            uvw = self._uvw(N)*lam
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                # Simulate and grid a single off-centre point source,
                # then shift back.
                vis = simulate_point(uvw, x/lam-dl, y/lam-dm)
                vis = visibility_shift(uvw, vis, dl, dm)
                # Should give us a point where we originally placed it
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvw/lam, vis)
                img = numpy.real(ifft(a))
                self.assertAlmostEqual(img[N//2+y,N//2+x], 1)

    def test_grid_transform(self):
        lam = 100
        s = 1 / numpy.sqrt(2)
        Ts = [ numpy.array([[1,0], [0,1]]),
               numpy.array([[-1,0], [0,1]]),
               numpy.array([[s,-s], [s,s]]),
               numpy.array([[2,0], [0,3]]),
               numpy.array([[1,2], [2,1]]) ]
        for T in Ts:
          # Invert transformation matrix
          Ti = numpy.linalg.inv(T)
          for N in range(3,7):
            # We will grid after the transformation. To make this
            # lossless we need to choose UVW such that they map
            # exactly to grid points *after* the transformation. We
            # can easily achieve this using the inverse transform.
            uvwt = self._uvw(N)*lam
            uvw = uvw_transform(uvwt, Ti)
            assert_allclose(uvw_transform(uvw, T), uvwt)
            xys = range(-(N//2),(N+1)//2)
            for xt, yt in itertools.product(xys, xys):
                # Same goes for grid positions: Determine position
                # before transformation such that we end up with a
                # point at x,y afterwards.
                x, y = numpy.dot([xt,yt], Ti)
                assert_allclose(numpy.dot([x,y], T), [xt,yt])
                # Now simulate at (x/y) using uvw, then grid using
                # the transformed uvwt, and the point should be at (xt/yt).
                vis = simulate_point(uvw, x/lam, y/lam)
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvwt/lam, vis)
                img = numpy.real(ifft(a))
                self.assertAlmostEqual(img[N//2+yt,N//2+xt], 1)

    def test_slice_vis(self):
        for N in range(2,10):
            for step in range(1,10):
                cs = self._uvw(N)
                slices = slice_vis(step, cs)
                assert_allclose(cs, numpy.vstack(slices))

    def test_grid_degrid_w(self):
        lam = 1000
        for uw, vw in [(.5,0),(0,.5),(-1,0),(0,-1)]:
          for N in range(1,6):
            # Generate UVW with w != 0, generate a w-kernel for every
            # unique w-value (should be exactly N by choice of uw,vw)
            uvw_all = self._uvw(N, uw, vw) * lam
            uvw_slices = slice_vis(N, sort_vis_w(uvw_all))
            # Generate kernels for every w-value, using the same far
            # field size and support as the grid size (perfect sampling).
            gcfs = [ w_kernel(N/lam, numpy.mean(uvw[:,2]), N, N, 1)
                     for uvw in uvw_slices ]
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                # Generate expected image for degridding
                img_ref = numpy.zeros((N, N), dtype=float)
                img_ref[N//2+y,N//2+x] = 1
                # Gridding does not have proper border handling, so we
                # need to artificially increase our grid size here by
                # duplicating grid data.
                a_ref = numpy.fft.ifftshift(fft(img_ref))
                a_ref = numpy.vstack(2*[numpy.hstack(2*[a_ref])])
                # Make grid for gridding
                a = numpy.zeros((2*N, 2*N), dtype=complex)
                for uvw, gcf in zip(uvw_slices, gcfs):
                    # Degridding result should match direct fourier transform
                    vis = simulate_point(uvw, x/lam, y/lam)
                    vis_d = convdegrid(gcf, a_ref, uvw/lam/2)
                    assert_allclose(vis, vis_d, rtol=0.1)
                    # Grid
                    convgrid(numpy.conj(gcf), a, uvw/lam/2, vis)
                # FFT halved generated grid (see above)
                a = numpy.fft.fftshift(a[:N,:N]+a[:N,N:]+a[N:,:N]+a[N:,N:])
                img = numpy.real(ifft(a))
                # Peak should be there, rest might be less precise as
                # we're not sampling the same w-plane any more
                assert_allclose(img[N//2+y,N//2+x], 1)
                assert_allclose(img, img_ref, atol=2e-3)

if __name__ == '__main__':
    unittest.main()
