import matplotlib.pyplot as plt
from math import pi, sqrt
import numpy as np

r2d=180.0/pi
r2a=r2d*3600.0

npts=100
alpha_resolution=5
alpha_FOV=3.0

# Set up thetares and derive thetafov and npix
thetamin=1.0e-2/r2a
thetamax=120/r2a
thetainc=(np.log10(thetamax)-np.log10(thetamin))/float(npts)
thetares=thetamin*np.power(10.0, thetainc*np.arange(npts))
# npix just depends on field of view, resolution, and the number of points per beam.
# fov should be that required to subtract all confusing sources - usually several times 
# the width of the first sidelobe
thetafov=alpha_FOV*np.sqrt(thetares)
npix=alpha_resolution*thetafov/thetares

# Now plot the fundamental curve - fov vs resolution. Denote region where w can be ignored.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(r2a*thetares, r2d*thetafov, 'b')
plt.xlim(r2a*thetamax*3.0, r2a*thetamin/3.0)
plt.ylim(3e-2, 5e1)
xmin=plt.xlim()[0]*np.ones(npts)
xmax=plt.xlim()[1]*np.ones(npts)
ymin=plt.ylim()[0]*np.ones(npts)
ymax=plt.ylim()[1]*np.ones(npts)

plt.xlabel("Resolution (arcsec)")
plt.ylabel("Field of view (degrees)")

# Plot number of pixels to span Fresnel zone
for maxnpix in [1e1,1e2,1e3,1e4,1e5,1e6,1e7]:
	plt.plot(r2a*thetares, r2d*maxnpix*thetares/alpha_resolution, 'g--')
	if (r2d*maxnpix*thetares[0]/alpha_resolution) > ymin[0]:
		ax.annotate("npix=%g" % maxnpix, [r2a*thetares[0], r2d*maxnpix*thetares[0]/alpha_resolution], horizontalalignment='left', verticalalignment='top',color="green")
	else:
		ax.annotate("npix=%g" % maxnpix, [3.0*r2a*thetafov[0]*alpha_FOV/maxnpix, 0.036], horizontalalignment='left', verticalalignment='top',color="green")
	
	
# Plot work load curves
#for R_F in [0.01,0.01,0.1,1,10,100,1000]:
for R_F in [0.01,0.1,1,10,100,1000,10000]:

	FOV=alpha_FOV*np.sqrt(thetares*R_F)
	if R_F == 1:
		ax.loglog(r2a*thetares, r2d*FOV, 'b', color='blue')
	else:
		ax.loglog(r2a*thetares, r2d*FOV, 'b--', color='blue')
	ax.annotate("R_F=%.2f" % R_F, [r2a*thetares[npts-1], r2d*FOV[npts-1]], horizontalalignment='right', verticalalignment='center',color="b")
		
# Annotate various projects

# SKA1_Dish 0.4GHz to 2GHz, 15m diameter, 150km
wave = [0.3/0.4, 0.3/2]
d=15.0
b=1.2e5
ax.plot([r2a*wave[0]/b, r2a*wave[1]/b], [r2d*wave[0]/d, r2d*wave[1]/d],  'r-',linewidth=2, label='SKA1_MID')
# SKA1_AA 0.05 to 0.35GHz 40m diameter, 80km
wave = [0.3/0.05, 0.3/0.35]
d=35.0
b=8e4
ax.plot([r2a*wave[0]/b, r2a*wave[1]/b], [r2d*wave[0]/d, r2d*wave[1]/d],  'pink',linewidth=2, label='SKA1_LOW')

# Now add the title, plot, and save hardcopy
plt.title("Fresnel number (blue). Number of pixels per axis (green)")
plt.text(0.8*xmin[0], 1.5*ymin[0], "alpha FOV = %.1f  " % alpha_FOV, horizontalalignment='left',
		 verticalalignment='bottom', color='black')
plt.text(0.8*xmin[0], 1.5*ymin[0], "alpha resolution = %.1f  " % alpha_resolution, horizontalalignment='left',
		 verticalalignment='top', color='black')
plt.legend()
plt.show()

plt.savefig("wska.pdf")
