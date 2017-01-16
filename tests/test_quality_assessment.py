"""Unit tests for quality assessment

realtimcornwell@gmail.com
"""
import unittest
import logging

from arl.data.data_models import QA

log = logging.getLogger(__name__)

class TestQualityAssessment(unittest.TestCase):
    
    def test_qa(self):
        qa=QA(origin='foo', data={'rms':100.0, 'median':10.0}, context='test of qa')
        log.debug(qa)

    def test_qa_gains(self):
        pass
    

    def test_qa_visibility(self):
        pass


    def test_qa_image(self):
        pass

if __name__ == '__main__':
    rootLog = logging.getLogger()
    rootLog.setLevel(logging.DEBUG)
    rootLog.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
