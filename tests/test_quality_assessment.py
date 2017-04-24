"""Unit tests for quality assessment


"""
import unittest
import logging

from arl.data.data_models import QA

log = logging.getLogger(__name__)

class TestQualityAssessment(unittest.TestCase):
    
    def test_qa(self):
        qa=QA(origin='foo', data={'rms':100.0, 'median':10.0}, context='test of qa')
        log.debug(str(qa))

if __name__ == '__main__':
    unittest.main()
