.. Calibration

.. toctree::
   :maxdepth: 3


Calibration
***********

Calibration is performed by fitting observed visibilities to a model visibility. The least squares fit algorithm uses
   an iterative substitution (or relaxation) algorithm from Larry D'Addario in the late seventies.

The equation to be minimised is:

.. math:: S = \sum_{t,f}^{}{\sum_{i,j}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{obs}} - J_{i}{J_{j}^{*}V}_{t,f,i,j}^{\text{mod}} \right|}^{2}}

Calibration control is via the :py:mod:`arl.calibration.calibration_context`. This supports the following Jones
   matrices::

   . T - Atmospheric phase
   . G - Electronics gain
   . P - Polarisation
   . B - Bandpass
   . I - Ionosphere

This is specified via a dictionary::

    contexts = {'T': {'shape': 'scalar', 'timeslice': 'auto', 'phase_only': True, 'first_iteration': 0},
                'G': {'shape': 'vector', 'timeslice': 60.0, 'phase_only': False, 'first_iteration': 0},
                'P': {'shape': 'matrix', 'timeslice': 1e4, 'phase_only': False, 'first_iteration': 0},
                'B': {'shape': 'vector', 'timeslice': 1e5, 'phase_only': False, 'first_iteration': 0},
                'I': {'shape': 'vector', 'timeslice': 1.0, 'phase_only': True, 'first_iteration': 0}}

Model Partition Calibration
***************************

The Model Partition Calibration approach is described in SDP memo 97.