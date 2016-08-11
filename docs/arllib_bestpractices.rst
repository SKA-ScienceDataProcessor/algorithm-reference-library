.. ARL documentation master file

.. toctree::
   :name: mastertoc
   :maxdepth: 2


:index:`Best Practices`
***********************

Design
------

The ARL has been designed in line with the following principles:

+ Data are held in Classes
+ The Data Classes correspond to familiar concepts in radio astronomy packages e.g. visibility, gaintable, image
+ The data members of the Data Classes are directly accessible by name e.g. .data, .name. .phasecentre
+ There are no methods attached to the data classes apart from variant constructors as needed.
+ Standalone, stateless functions are used for all processing
+ All function parameters are passed by the kwargs mechanism
