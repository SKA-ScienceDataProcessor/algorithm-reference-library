""" Unit tests for json helpers


"""
import logging
import unittest

from data_models.parameters import arl_path
from external_interface.arl_json.json_helpers import json_to_linspace, json_to_quantity, json_to_skycoord
from external_interface.execution_helper import initialise_config_wrapper

log = logging.getLogger(__name__)


class TestJSONHelpers(unittest.TestCase):
    def setUp(self):
        self.config = initialise_config_wrapper(arl_path("tests/workflows/test_json_helpers.json"))
    
    def test_json_linspace(self):
        f = {"start": 0.9e8, "stop": 1.1e8, "steps": 7}
        ls = json_to_linspace(f)
        assert ls[-1] == 1.1e8
        assert ls[0] == 0.9e8
        assert len(ls) == 7
        
        f = {"start": 0.9e8, "stop": 1.1e8, "steps": 0}
        ls = json_to_linspace(f)
        assert len(ls) == 0
        
        with self.assertRaises(ValueError) as context:
            f = {"start": 0.9e8, "stop": 1.1e8, "steps": "foo"}
            ls = json_to_linspace(f)
    
    def test_json_quantity(self):
        q = {"value": 30.0, "unit": "deg"}
        quant = json_to_quantity(q)
        with self.assertRaises(AssertionError) as context:
            q = {"value": 30.0, "unit": 0.0}
            quant = json_to_quantity(q)
        with self.assertRaises(ValueError) as context:
            q = {"value": "foo", "unit": 0.0}
            quant = json_to_quantity(q)
    
    def test_json_skycoord(self):
        d = {
            "ra": {
                "value": 30.0,
                "unit": "deg"
            },
            "dec": {
                "value": -60.0,
                "unit": "deg"
            },
            "frame": "icrs",
            "equinox": "j2000"
        }
        direction = json_to_skycoord(d)
        with self.assertRaises(KeyError) as context:
            dt = {
                "ra": {
                    "value": 30.0,
                    "unit": "deg"
                }
            }
            direction = json_to_skycoord(dt)
        with self.assertRaises(KeyError) as context:
            dt = {
                "ra": {
                    "value": 30.0,
                    "unit": "deg"
                },
                "dec": {
                    "value": -60.0,
                    "unit": "deg"
                },
                "frame": "icrs",
            }
            direction = json_to_skycoord(dt)


if __name__ == '__main__':
    unittest.main()
