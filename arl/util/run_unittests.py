
#
"""
Definition of structures needed by the function interface.
"""
import sys
import unittest


import logging
log = logging.getLogger(__name__)


def run_unittests(logLevel=logging.DEBUG, *args, **kwargs):
    """Runs the unit tests in all loaded modules.

    :param logLevel: The amount of logging to generate. By default, we
      show all log messages (level DEBUG)
    """

    # Set up logging environment
    rootLog = logging.getLogger()
    rootLog.setLevel(logLevel)
    rootLog.addHandler(logging.StreamHandler(sys.stderr))

    # Call unittest main
    unittest.main(*args, **kwargs)
