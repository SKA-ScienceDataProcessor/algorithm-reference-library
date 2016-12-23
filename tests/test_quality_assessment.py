"""Unit tests for quality assessment

realtimcornwell@gmail.com
"""
import unittest

from arl.data.data_models import QA


class TestQualityAssessment(unittest.TestCase):
    
    def test_qa(self):
        qa=QA(origin='foo', data={'rms':100.0, 'median':10.0}, context='test of qa')
        print(qa)

    def test_qa_gains(self):
        pass
    

    def test_qa_visibility(self):
        pass


    def test_qa_image(self):
        pass

if __name__ == '__main__':
    unittest.main()
