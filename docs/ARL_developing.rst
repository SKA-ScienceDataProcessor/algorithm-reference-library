
Best Practices
**************

Coding and documentation
========================

Use the `SIP Coding and Documentation Guide <https://confluence.ska-sdp
.org/display/SIP/Coding+and+Documentation+Guide+for+SIP/>`_ .

We recommend using a tool to help ensure PEP 8 compliance. PyCharm does a good job at this and other code quality
checks.

Design
======

The ARL has been designed in line with the following principles:

+ Data are held in Classes.
+ The Data Classes correspond to familiar concepts in radio astronomy packages e.g. visibility, gaintable, image.
+ The data members of the Data Classes are directly accessible by name e.g. .data, .name. .phasecentre.
+ Direct access to the data members is envisaged.
+ There are no methods attached to the data classes apart from variant constructors as needed.
+ Standalone, stateless functions are used for all processing.

Additions and changes should adhere to these principles.

Submitting code
===============

ARL is hosted on the `SDP github <https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library.git>`_ .

We are open to pull requests submitted via github.