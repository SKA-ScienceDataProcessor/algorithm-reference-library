# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>

"""Coordinate support

USEFUL NOTES
---------------------------------------------------------------------------------

From http://casa.nrao.edu/Memos/CoordConvention.pdf :

UVW is a right-handed coordinate system, with W pointing towards the
source, and a baseline convention of :math:`ant2 - ant1` where
:math:`index(ant1) < index(ant2)`.  Consider an XYZ Celestial
coordinate system centered at the location of the interferometer, with
:math:`X` towards the East, :math:`Z` towards the NCP and :math:`Y` to
complete a right-handed system. The UVW coordinate system is then
defined by the hour-angle and declination of the phase-reference
direction such that

1. when the direction of observation is the NCP (`ha=0,dec=90`),
   the UVW coordinates are aligned with XYZ,

2. V, W and the NCP are always on a Great circle,

3. when W is on the local meridian, U points East

4. when the direction of observation is at zero declination, an
   hour-angle of -6 hours makes W point due East.

The :math:`(l,m,n)` coordinates are parallel to :math:`(u,v,w)` such
that :math:`l` increases with Right-Ascension (or increasing longitude
coordinate), :math:`m` increases with Declination, and :math:`n` is
towards the source. With this convention, images will have Right
Ascension increasing from Right to Left, and Declination increasing
from Bottom to Top.

Changes
---------------------------------------------------------------------------------

[150323 - AMS]    Correction to rot(), introduced -1 factor in v calculation.
                  Change to bsl(), to agree with casa convention.

"""

import numpy
from astropy.coordinates import SkyCoord, CartesianRepresentation


# ---------------------------------------------------------------------------------

def xyz_at_latitude(local_xyz, lat):
    """
    Rotate local XYZ coordinates into celestial XYZ coordinates. These
    coordinate systems are very similar, with X pointing towards the
    geographical east in both cases. However, before the rotation Z
    points towards the zenith, whereas afterwards it will point towards
    celestial north (parallel to the earth axis).

    :param lat: target latitude (radians or astropy quantity)
    :param local_xyz: Array of local XYZ coordinates
    :returns: Celestial XYZ coordinates
    """
    
    x, y, z = numpy.hsplit(local_xyz, 3)
    
    lat2 = numpy.pi / 2 - lat
    y2 = -z * numpy.sin(lat2) + y * numpy.cos(lat2)
    z2 = z * numpy.cos(lat2) + y * numpy.sin(lat2)
    
    return numpy.hstack([x, y2, z2])


# ---------------------------------------------------------------------------------

def xyz_to_uvw(xyz, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions in earth coordinates to
    :math:`(u,v,w)` coordinates relative to astronomical source
    position :math:`(ha, dec)`. Can be used for both antenna positions
    as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre.
    """
    
    x, y, z = numpy.hsplit(xyz, 3)
    
    # Two rotations:
    #  1. by 'ha' along the z axis
    #  2. by '90-dec' along the u axis
    u = x * numpy.cos(ha) - y * numpy.sin(ha)
    v0 = x * numpy.sin(ha) + y * numpy.cos(ha)
    w = z * numpy.sin(dec) - v0 * numpy.cos(dec)
    v = z * numpy.cos(dec) + v0 * numpy.sin(dec)
    
    return numpy.hstack([u, v, w])


# ---------------------------------------------------------------------------------

def uvw_to_xyz(uvw, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions relative to a sky position at
    :math:`(ha, dec)` to earth coordinates. Can be used for both
    antenna positions as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param uvw: :math:`(u,v,w)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre
    """
    
    u, v, w = numpy.hsplit(uvw, 3)
    
    # Two rotations:
    #  1. by 'dec-90' along the u axis
    #  2. by '-ha' along the z axis
    v0 = v * numpy.sin(dec) - w * numpy.cos(dec)
    z = v * numpy.cos(dec) + w * numpy.sin(dec)
    x = u * numpy.cos(ha) + v0 * numpy.sin(ha)
    y = -u * numpy.sin(ha) + v0 * numpy.cos(ha)
    
    return numpy.hstack([x, y, z])


# ---------------------------------------------------------------------------------

def baselines(ants_uvw):
    """
    Compute baselines in uvw co-ordinate system from
    uvw co-ordinate system station positions

    :param ants_uvw: `(u,v,w)` co-ordinates of antennas in array
    """
    
    res = []
    for i in range(ants_uvw.shape[0]):
        for j in range(i + 1, ants_uvw.shape[0]):
            res.append(ants_uvw[j] - ants_uvw[i])
    
    basel_uvw = numpy.array(res)
    
    return basel_uvw


# ---------------------------------------------------------------------------------

def xyz_to_baselines(ants_xyz, ha_range, dec):
    """
    Calculate baselines in :math:`(u,v,w)` co-ordinate system
    for a range of hour angles (i.e. non-snapshot observation)
    to create a uvw sampling distribution

    :param ants_xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha_range: list of hour angle values for astronomical source as function of time
    :param dec: declination of astronomical source [constant, not :math:`f(t)`]
    """
    
    dist_uvw = numpy.concatenate([baselines(xyz_to_uvw(ants_xyz, hax, dec)) for hax in ha_range])
    return dist_uvw


# ---------------------------------------------------------------------------------

def skycoord_to_lmn(pos: SkyCoord, phasecentre: SkyCoord):
    """
    Convert astropy sky coordinates into the l,m,n coordinate system
    relative to a phase centre.

    The l,m,n is a RHS coordinate system with
    * its origin on the sky sphere
    * m,n and the celestial north on the same plane
    * l,m a tangential plane of the sky sphere

    Note that this means that l increases east-wards
    """
    
    # Determine relative sky position
    todc = pos.transform_to(phasecentre.skyoffset_frame())
    dc = todc.represent_as(CartesianRepresentation)
    
    # Do coordinate transformation - astropy's relative coordinates do
    # not quite follow imaging conventions
    return dc.y.value, dc.z.value, dc.x.value - 1


def simulate_point(dist_uvw, l, m):
    """
    Simulate visibilities for unit amplitude point source at
    direction cosines (l,m) relative to the phase centre.

    This includes phase tracking to the centre of the field (hence the minus 1
    in the exponent.)

    Note that point source is delta function, therefore the
    FT relationship becomes an exponential, evaluated at
    (uvw.lmn)

    :param dist_uvw: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param l: horizontal direction cosine relative to phase tracking centre
    :param m: orthogonal directon cosine relative to phase tracking centre
    """
    
    # vector direction to source
    s = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])
    # complex valued Visibility data
    return numpy.exp(-2j * numpy.pi * numpy.dot(dist_uvw, s))


def visibility_shift(uvw, vis, dl, dm):
    """
    Shift visibilities by the given image-space distance. This is
    based on simple FFT laws. It will require kernels to be suitably
    shifted as well to work correctly.

    :param uvw:
    :param vis: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param vis: Input visibilities
    :param dl: Horizontal shift distance as directional cosine
    :param dm: Vertical shift distance as directional cosine
    :returns: New visibilities

    """
    
    s = numpy.array([dl, dm])
    return vis * numpy.exp(-2j * numpy.pi * numpy.dot(uvw[:, 0:2], s))


def uvw_transform(uvw, transform_matrix):
    """
    Transforms UVW baseline coordinates such that the image is
    transformed with the given matrix. Will require kernels to be
    suitably transformed to work correctly.

    Reference: Sault, R. J., L. Staveley-Smith, and W. N. Brouw. "An
    approach to interferometric mosaicing." Astronomy and Astrophysics
    Supplement Series 120 (1996): 375-384.

    :param uvw: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param transform_matrix: 2x2 matrix for image transformation
    :returns: New baseline coordinates
    """
    
    # Calculate transformation matrix (see Sault)
    tt = numpy.linalg.inv(numpy.transpose(transform_matrix))
    # Apply to uv coordinates
    uv1 = numpy.dot(uvw[:, 0:2], transform_matrix)
    # Restack with original w values
    return numpy.hstack([uv1, uvw[:, 2:3]])
