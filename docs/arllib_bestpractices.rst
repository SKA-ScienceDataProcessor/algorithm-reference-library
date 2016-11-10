.. ARL best practices file

.. toctree::
   :name: mastertoc
   :maxdepth: 2


Best Practices
**************


Coding and documentation
------------------------

Use the `SIP Coding and Documentation Guide <https://confluence.ska-sdp
.org/display/SIP/Coding+and+Documentation+Guide+for+SIP/>`_ .

Design
------

The ARL has been designed in line with the following principles:

+ Data are held in Classes.
+ The Data Classes correspond to familiar concepts in radio astronomy packages e.g. visibility, gaintable, image.
+ The data members of the Data Classes are directly accessible by name e.g. .data, .name. .phasecentre.
+ Direct access to the data members is envisaged.
+ There are no methods attached to the data classes apart from variant constructors as needed.
+ Standalone, stateless functions are used for all processing.
+ All function parameters are passed by the kwargs mechanism.

Additions and changes should adhere to these principles.

Naming
------

* Names should obey the `SIP Coding and Documentation Guide <https://confluence.ska-sdp.org/display/SIP/Coding+and+Documentation+Guide+for+SIP/>`_ guide.
* For functions that move data in and out of ARL, use import/export.
* For functions that provide persistent, use read/save.

