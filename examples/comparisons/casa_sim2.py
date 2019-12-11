
im.open('sim-2.ms')
im.defineimage(cellx='20arcsec', nx=512)
im.makeimage('sim-2.dirty')
im.makeimage('observed', 'casa_imaging_sim_2_dirty')
ia.open('casa_imaging_sim_2_dirty')
ia.tofits('casa_imaging_sim_2_dirty.fits')

