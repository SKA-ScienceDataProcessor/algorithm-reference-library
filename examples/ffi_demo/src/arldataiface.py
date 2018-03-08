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
    for i in range(s):  b.PrependFloat64(wcs.crval[i])
    crval=b.EndVector(s)

    arlmd.ImageWCS.ImageWCSStart(b)
    arlmd.ImageWCS.ImageWCSAddCrpix(b, crpix=crpix)
    arlmd.ImageWCS.ImageWCSAddCrval(b, crval=crval)    
    img=arlmd.ImageWCS.ImageWCSEnd(b)
    b.Finish(img)

    return b.Bytes[b.Head():]




