.. Background

.. toctree::
   :maxdepth: 3

Background
==========

Core motivations
----------------

- In many software packages, the only function specification is the application code itself. Although the underlying
  algorithm may be published, the implementation tends to diverge over time, making this method of
  documentation less effective. The algorithm reference library is designed to present imaging algorithms in a simple
  Python-based form. This is so that the implemented functions can be seen and understood without resorting to
  interpreting source code shaped by real-world concerns such as optimisations.

- Maintenance of the reference library over time is a choice for operations and we do not discuss it further here.

- Desire for simple test version: for example, scientists may wish to understand how the algorithm works and see it
  tested in particular circumstances. Or a software developer wish to compare it to production code.

Purpose
-------

- Documentation: The primary purpose of the library is to be easily understandable to people not familiar with radio
  interferometry imaging. This means that the library should be broken down into a number of small, well-documented
  functions. Aside from the code itself, these functions will be further explained by documentation as well as material
  demonstrating its usage. Where such efforts would impact the clarity of the code itself it should be kept separate
  (e.g. example notebooks).

- Testbed for experimentation: One purpose for the library is to facilitate experimentation with the algorithm
  without touching the production code. Production code may be specialised due to the need for optimization, however
  the reference implementation should avoid any assumptions not actually  from the theory of interferometry imaging.

- Publication e.g. via github: All algorithms used in production code should be known and published. If the
  algorithms are available separately from the production code then others can make use of the published code for small
  projects or to start on an improved algorithm.

- Conduit for algorithms into SKA: The library can serve as a conduit for algorithms into the SKA production system.
  A scientist can provide Python Version of an algorithm which then can be translated into optimized production code by
  the SKA computer team.

- Algorithm unaffected by optimization: Production code is likely to be obscured by the need to optimize in various
  ways. The algorithms in the library will avoid this as much as possible in order to remain clear and transparent.
  Where algorithms need to be optimised in order to remain executable on typical hardware, we might opt for providing
  multiple equivalent algorithm variants.

- Deliver algorithms for construction phase: The algorithm reference library Will also serve as a resource for the
  delivery of algorithms to the construction phase. It is likely that much of the production code will be written by
  people not intimately familiar with radio astronomy. Experience shows that such developers can often work from a
  simple example of the algorithm.

- Reference for results: The library will also serve to provide reference results for the production code. This is
  not entirely straightforward because the algorithms in both cases work in different contexts. Code that establishes
  interoperability with external code will have to kept separate to not clutter the core implementation.  This means
  that we will not be able to guarantee comparability in all cases. In that case, it will be the responsibility other
  developers of the production code to establish it - for example by using suitably reduced data sets.

Stakeholders
------------

- SDP design team: The principal stakeholders for the algorithm reference library are the SDP Design Team. They will
  benefit from having cleared descriptions of algorithms for all activities such as resource estimation, parameter
  setting, definition of pipelines, and so on.

- SKA Project Scientists: The SKA project scientists must be able to understand the algorithms used in the pipelines.
  This is essential if they are going to be assured that the processing is as desired, and relay that to the observers.

- External scientists: External scientists and observers using the telescope will benefit into ways. First, in
  understanding the processing taking place in the pipelines, and second, being able to bring new algorithms for
  deployment into the pipelines.

- SDP contractors: Depending upon the procurement model, SDP may be developed by a team without very much domain
  knowledge. While expect the documentation of the entire system to be in good shape after CDR, the algorithms are the
  very core of the system I must be communicated clearly and concisely.  We can expect that any possible contractors
  considering a bid would be reassured by the presence of algorithm reference library.

- Outreach: Finally, outreach may be a consumer of the library. For example, the library could be made available
  to students at various levels to introduce them to astronomical data-processing concepts.

Prior art
---------

  LAPACK is an example of a library that mutated into a reference library. The original code was written in
  straightforward FORTRAN  but now many variants have been spawned including for example Versions optimized for
  particular hardware, or using software scheduling techniques such as DAGs to arrange their internal processing.  The
  optimized variants must always agree with the reference code.

Requirements
------------

- Minimal implementation: The implementation should be minimal making use of as few external libraries as possible.
  Python is a good choice for the implementation because the associated libraries are powerful and well-defined.

- Use numpy whenever possible: Some form of numeric processing is inevitably necessary. There is also need for
  efficient bulk data transfer between functions. For consistency, we choose to adopt the numpy library for both
  algorithm and interface definition.

- Take algorithms with established provenance: While the purpose of the library is to define the algorithms clearly,
  the algorithms themselves should have well-defined provenance. Acceptable forms of provenance include publication in a
  peer-reviewed journal, publication in a well-defined memo series, and use in a well-defined production system.  In
  time we might expect that the algorithm reference library will itself provide sufficient provenance. This depends
  upon the processes to maintain the library being stringently defined and applied.

- No optimization: No optimization should be performed on algorithms in the library if doing so obscures the
  fundamentals of the algorithm.  Runtime of the testsuite should not be consideration except in so far as it prevents
  effective use.

- V&V begins here: Validation and verification of the pipeline processing begins in the algorithm reference library.
  That means that it should be held to high standards of submission, testing, curation, and documentation.

- Single threaded: All algorithms should be single threaded unless multithreading is absolutely required to  achieve
  acceptable performance. However, as distributed execution is going to be vital for the SDP, special take should be
  taken to document and demonstrate parallelism opportunities.

- Memory limit: The memory used should be compatible with execution on a personal computer or laptop.

Algorithms to be defined
------------------------

The following list gives an initial set of algorithms to be defined. It is more important to have the overall
framework in place expeditiously than to have each algorithm be state-of-the-art.

   - Simulation

      - Station/Antenna locations
      - Illumination/Primary beam models
      - Generation of visibility data
      - Generation of gain tables

   - Calibration

      - Calibration solvers

         - Stefcal

      - Calibration application

         - Gain interpolation
         - Gain application

      - Self-calibration

   - Visibility plane

      - Convolution kernels

         - Standard
         - W Projection
         - AW Projection
         - AWI Projection

      - Degridding/Gridding

         - 2D
         - W projection
         - W slices
         - W snapshots

      - Preconditioning/Weighting

         - Uniform
         - Briggs

   - Visibility plane to/from Image plane

      - DFT
      - Faceting
      - Phase rotation
      - Averaging/deaveraging
      - Major cycles

   - Image plane

      - Source finding
      - Source fitting
      - Reprojection
      - Interpolation
      - MSClean minor cycle (for spectral line)
      - MSMFS minor cycle (for continuum)


To test and demonstrate completeness, the main pipelines will be implemented.

Testing
-------

- Testing philosophy: The essence of an algorithm reference library is that it should be used as the standard for
  the structure and execution of a particular algorithm.  This can only be done if the algorithm and the associated
  code are tested exhaustively.

- We will use three ways of performing testing of the code

  - Unit tests of all functions:
  - Regression tests of the complete algorithm over a complete set of inputs.
  - Code reviews (either single person or group read-throughs).

- Test suite via Jenkins: The algorithm reference library will therefore come with a complete set of unit tests and
  regression tests. These should be run automatically, by, for example, a framework such as Jenkins, on any change to
  ensure their errors are caught quickly and not compounded.
