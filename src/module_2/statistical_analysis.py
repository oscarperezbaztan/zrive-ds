
from scipy import stats
from typing import Union
import pandas as pd

def test_normality_shapiro(data: pd.Series, alpha: float = 0.05) -> tuple[float, bool]:

    shapiro_test = stats.shapiro(data)
    p_value = shapiro_test.pvalue
    reject_null_hypothesis = p_value < alpha
    return p_value, reject_null_hypothesis


def t_student_test(column1, column2):

    t_statistic, p_value = stats.ttest_ind(column1, column2)

    return t_statistic, p_value


def mann_whitney_test(column1, column2):

    u_statistic, p_value = stats.mannwhitneyu(column1, column2)
    return u_statistic, p_value

