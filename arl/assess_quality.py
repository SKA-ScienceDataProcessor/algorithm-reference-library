# Tim Cornwell <realtimcornwell@gmail.com>
#
# Visibility data structure: a Table with columns ['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight']
# and an attached attribute which is the frequency of each channel

from astropy import constants as const
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.table import Table, vstack

from crocodile.simulate import *

from arl.simulate_visibility import create_named_configuration
from arl.data_models import *

"""
Holder for the Quality Assessment
The data structure:
- TBC
"""

        
def export_AQ(aq: AQ):
    """Export the accumulate QA info
        
    """
    print("assess_quality.export_AQ: not yet implemented")
