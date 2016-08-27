# Tim Cornwell <realtimcornwell@gmail.com>
#
#

from arl.visibility_calibration import solve_gains, qa_gaintable, create_visibility
from arl.fourier_transforms import predict_visibility, invert_visibility, combine_visibility

def RCAL(parameters, logfile=None):
    """ Real-time calibration
    
    """
    print(parameters)
    vt = parameters['RCAL']['visibility']
    sm = parameters['RCAL']['skymodel']
    
    gains = solve_gains(vt, sm, **parameters)
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.RCAL: Solution good")
    return qa


def continuum_imaging(parameters, logfile=None):
    
    kwargs=parameters['continuum_imaging']
    
    nmajor = kwargs.get('nmajor', 5)
    deconvolver=kwargs.get('deconvolver')
    sm = kwargs.get("skymodel")
    
    print("pipelines.continuum_imaging: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vtpred = predict_visibility(vt, sm, **kwargs)
    vtres = combine_visibility(vt, vtpred, 1.0, -1.0)
    dirty, psf, sumwt = invert_visibility(vtres, **kwargs)
    thresh = kwargs.get("threshold", 0.0)
    
    comp = sm.images[0]
    for i in range(nmajor):
        print("pipelines.continuum_imaging.solve_skymodel: Start of major cycle %d" % i)
        cc, res = deconvolver(dirty, psf, **kwargs)
        comp += cc
        vtpred = predict_visibility(vt, sm, **kwargs)
        vtres = combine_visibility(vt, vtpred, 1.0, -1.0)
        dirty, psf, sumwt = invert_visibility(vtres, **kwargs)
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            print("pipelines.continuum_imaging.solve_skymodel: Reached stopping threshold %.6f Jy" % thresh)
            break
        print("pipelines.continuum_imaging.solve_skymodel: End of major cycle")
    print("pipelines.continuum_imaging.solve_skymodel: End of major cycles")
    return vtres, sm

if __name__ == '__main__':
    import numpy
    from arl.test_support import create_named_configuration
    from astropy.coordinates import SkyCoord
    config = create_named_configuration('VLAA')
    times = numpy.arange(-3.0, 0.0, 3.0 / 60.0) * numpy.pi / 12.0
    freq = numpy.arange(5e6, 150.0e6, 1e7)
    phasecentre = SkyCoord(ra='00h42m30s', dec='+41d12m00s', frame='icrs', equinox=2000.0)
    vt = create_visibility(config, times, freq, weight=1.0, phasecentre=phasecentre)
    parameters={}
    parameters={'RCAL':{'visibility':vt, 'skymodel':None}}
    rcal=RCAL(parameters)
        
            