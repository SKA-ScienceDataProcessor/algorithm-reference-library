.. Calibration

.. toctree::
   :maxdepth: 3


Calibration
***********

Calibration is performed by fitting observed visibilities to a model visibility. The least squares fit algorithm uses
 an iterative substitution (or relaxation) algorithm from Larry D'Addario in the late seventies.

Serial Calibration
==================

Serial calibration is straighforward. The equation to be minimised is:

.. math:: S = \sum_{t,f}^{}{\sum_{i,j}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{obs}} - J_{i}{J_{j}^{*}V}_{t,f,i,j}^{\text{mod}} \right|}^{2}}

Parallel Calibration
====================

Parallel calibration is more difficult if we wish to avoid sending large amounts of data to a central solver. ARL
uses a trick to reduce the data sent. The visibilities are converted to a point source equivalent by dividing by the
model visibility, and then averaged up to the solution interval. Only the averaged data then need to be sent to a
globasl solver, and only the gain tables need be sent back to be applied.

Let us consider the case where we have a lot of data (in baseline, time,
and frequency) spread over Compute Islands. We wish to perform a joint
minimisation to estimate Jones matrices.

The equation to be minimised is:

.. math:: S = \sum_{t,f}^{}{\sum_{i,j}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{obs}} - J_{i}{J_{j}^{*}V}_{t,f,i,j}^{\text{mod}} \right|}^{2}}

This involves data on the compute islands and global data (the Jones
matrices). Minimising\ :math:`\text{\ S}` would therefore seem to
require lots of data transfer to and from the Compute Islands for each
iteration of the solver. The data to be transmitted are:

.. math:: {V_{t,f,i,j}^{\text{obs}},V}_{t,f,i,j}^{\text{mod}},w_{t,f,i,j}

However, as indicated above, we can avoid some of this traffic by re-stating the problem.
:math:`S` can be rearranged as:

.. math:: S = \sum_{t,f}^{}{\sum_{i,j}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{mod}} \right|^{2}\left| X_{t,f,i,j}^{\text{obs}} - J_{i}J_{j}^{*} \right|^{2}}}

.. math:: S = \sum_{i,j}^{}{\sum_{t,f}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{mod}} \right|^{2}\left| X_{t,f,i,j}^{\text{obs}} - J_{i}J_{j}^{*} \right|^{2}}}

We therefore can write the misfit term as:

.. math:: S = \sum_{i,j}^{}{{\overline{w}}_{i,j}\left| {\overline{X}}_{i,j}^{\text{obs}} - J_{i}J_{j}^{*} \right|^{2}}

where:

.. math:: {\overline{w}}_{i,j} = \sum_{t,f}^{}{{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{mod}} \right|}^{2}\ }

.. math:: {\overline{X}}_{i,j} = \sum_{t,f}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{mod}} \right|^{2}}X_{t,f,i,j}^{\text{obs}}

This corresponds to a point source calibration using the data averaged
over time and frequency. The consequence is that the time-smearing
constraint that would limit solution using just the
:math:`V_{t,f,i,j}^{\text{obs}},\ V_{t,f,i,j}^{\text{mod}},w_{t,f,i,j}`
terms is much relaxed because :math:`{\overline{X}}_{i,j}^{\text{obs}}`,
:math:`{\overline{w}}_{i,j}` should vary less with time and frequency.

Now let us consider the solution, using dataflow Compute Islands
processing the visibility data, and a solver Compute Island dedicated to
solving.

-  Each dataflow island computes a contribution to
   :math:`{\overline{(X}}_{i,j,\ }{\overline{w}}_{i,j})` and sends it to
   the solver island.

-  The solver island solves for the Jones matrices: :math:`J_{i}` by
   minimising
   :math:`S = \sum_{i,j}^{}{{\overline{w}}_{i,j}\left| {\overline{X}}_{i,j}^{\text{obs}} - J_{i}J_{j}^{*} \right|^{2}}`

-  The solver sends the Jones matrices :math:`J_{i}`\ to the data flow
   CI’s and to the Telescope Model

-  The data flow CI’s apply the relevant correction.

This is quite straightforward.

This is for the simplest case of global Jones. In practice there are a
couple more cases:

+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
| **Type of solution**                   | **Compute Island tasks**                                                                                                                                                                            | **Solver Island tasks**                                                                                                                |
+========================================+=====================================================================================================================================================================================================+========================================================================================================================================+
| Independent :math:`J_{i,f}`            | For each :math:`f\ ,\ `\ solve :math:`S_{f} = \sum_{i,j}^{}{\sum_{t}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{mod}} \right|^{2}\left| X_{t,f,i,j}^{\text{obs}} - J_{i,f}J_{j,f}^{*} \right|^{2}}}`   | Not needed                                                                                                                             |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
| .. math:: J_{i}                        | 1. Compute :math:`{\overline{X}}_{i,j},{\overline{w}}_{i,j}`\ for local :math:`f`                                                                                                                   | Solve\ :math:`\ S = \sum_{i,j}^{}{{\overline{w}}_{i,j}\left| {\overline{X}}_{i,j}^{\text{obs}} - J_{i}J_{j}^{*} \right|^{2}}`          |
|                                        |                                                                                                                                                                                                     |                                                                                                                                        |
|                                        | 2. Transmit to solver island                                                                                                                                                                        |                                                                                                                                        |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
| :math:`J_{i,f}` polynomial or spline   | 1. Compute :math:`{\overline{X}}_{f,i,j},{\overline{w}}_{f,i,j}`                                                                                                                                    | Solve :math:`S = \sum_{f,i,j}^{}{{\overline{w}}_{f,i,j}\left| {\overline{X}}_{f,i,j}^{\text{obs}} - J_{i,f}J_{j,f}^{*} \right|^{2}}`   |
|                                        |                                                                                                                                                                                                     |                                                                                                                                        |
|                                        | 2. Transmit to solver island                                                                                                                                                                        |                                                                                                                                        |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+

The conclusion is that the global solution for the Jones matrices need
not require a lot of data transfer throughout the solution process.
There remains a global barrier point in the solver.

:download:`Global Calibration <theory/ARL_global_calibration.pdf>`.