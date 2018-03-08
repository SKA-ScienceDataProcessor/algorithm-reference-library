# Routines to assist with data-interfaces to ARL

import flatbuffers
# The meta-data modules. See $ARLHOME/metadata 
import arlmd, arlmd.ImageWCS

def ImageWCSFb(wcs):
    b=flatbuffers.Builder(0)
    s=wcs.crpix.shape[0]

    arlmd.ImageWCS.ImageWCSStartCrpixVector(b, numElems=s)
    for x in wcs.crpix: b.PrependFloat64(x)
    crpix=b.EndVector(s)

    arlmd.ImageWCS.ImageWCSStartCrpixVector(b, numElems=s)
    for x in wcs.crpix: b.PrependFloat64(x)
    crpix=b.EndVector(s)

    arlmd.ImageWCS.ImageWCSStartCrvalVector(b, numElems=s)
    for x in wcs.crval: b.PrependFloat64(x)
    crval=b.EndVector(s)

    arlmd.ImageWCS.ImageWCSStart(b)
    arlmd.ImageWCS.ImageWCSAddCrpix(b, crpix=crpix)
    arlmd.ImageWCS.ImageWCSAddCrval(b, crval=crval)    
    arlmd.ImageWCS.ImageWCSEnd(b)
    return b




