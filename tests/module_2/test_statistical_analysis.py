
import sys
sys.path.append('/home/oscar/zrive-ds')

import unittest
import numpy as np
import pandas as pd
from scipy.stats import anderson
from src.module_2.statistical_analysis import *

class TestNormalityShapiroFunction(unittest.TestCase):

    def test_normal_data(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='column_name')
        p_value, reject_null_hypothesis = test_normality_shapiro(data)
        self.assertGreater(p_value, 0.05)  
        self.assertFalse(reject_null_hypothesis)  

    def test_non_normal_data(self):
        data = pd.Series(np.random.random(100), name='column_name')  
        p_value, reject_null_hypothesis = test_normality_shapiro(data)
        self.assertLess(p_value, 0.05)  
        self.assertTrue(reject_null_hypothesis)  


class TestTStudentFunction(unittest.TestCase):

    def test_t_student(self):

        column1 = pd.Series([1, 2, 3, 4, 5])
        column2 = pd.Series([6, 7, 8, 9, 10])

        t_statistic, p_value = t_student_test(column1, column2)

        self.assertIsInstance(t_statistic, float)
        self.assertIsInstance(p_value, float)
        self.assertLessEqual(p_value, 1.0)  


class TestMannWhitneyFunction(unittest.TestCase):

    def test_mann_whitney(self):

        column1 = pd.Series([1, 2, 3, 4, 5])
        column2 = pd.Series([6, 7, 8, 9, 10])

        u_statistic, p_value = mann_whitney_test(column1, column2)

        self.assertIsInstance(u_statistic, float)
        self.assertIsInstance(p_value, float)
        self.assertLessEqual(p_value, 1.0)  


if __name__ == '__main__':
    unittest.main()


