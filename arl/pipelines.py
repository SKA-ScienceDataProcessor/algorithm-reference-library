# Tim Cornwell <realtimcornwell@gmail.com>
#
#

from arl.visibility_calibration import solve_gains, qa_gaintable, create_visibility

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
        
            