"""Definition of ARL JSON schema and helper functions

The ARL JSON schema contains various definitions. Some are for logging and execution, some are for standard uses (
such as image and imaging), some are helper definitions (like linspace), and some are directed for specific processing_component_interface.

Compliance with the schema is checked on loading. Definitions not present in the schema are allowed. Finally no
default handling is currently present so all fields must be defined.

"""