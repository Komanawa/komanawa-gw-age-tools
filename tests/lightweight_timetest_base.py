"""
created matt_dumont 
on: 16/05/24
"""
import numpy as np
from copy import deepcopy

import pandas as pd
import sys

sys.path.append('/home/matt_dumont/PycharmProjects/komanawa-gw-age-tools/src')
from komanawa.gw_age_tools import check_age_inputs, make_age_dist
from komanawa.gw_age_tools.lightweight import lightweight_predict_future, lightweight_v2_predict_future, \
    lightweight_v3_predict_future, lightweight_predict_future_int, lightweight_predict_future_int_np, lightweight_v2_predict_future_np

precision = 2
mrt, mrt_p1 = 31.5, 31.5
frac_p1 = 1
mrt_p2 = None
f_p1 = 0.65619
f_p2 = 0.65619  # dummy
mrt, mrt_p2 = check_age_inputs(mrt, mrt_p1, mrt_p2, frac_p1, precision, f_p1, f_p2)
age_step, ages, age_fractions = make_age_dist(mrt, mrt_p1, mrt_p2, frac_p1, precision, f_p1, f_p2)

source1 = pd.Series(index=np.arange(-ages.max(), 500, 10 ** -precision).round(precision), data=np.nan, dtype=float)
source1.iloc[0] = 1
source1.iloc[-1] = 10
source1 = source1.interpolate(method='index')
source1 = source1.sort_index()
source2 = deepcopy(source1)
source3 = deepcopy(source1)
source4 = deepcopy(source1)
source4.index = (np.round(source4.index * int(10 ** precision))).astype(int)
ages4 = (np.round(deepcopy(ages) * int(10 ** precision))).astype(int)

outages = np.linspace(1, 400, 1000)
outages4 = (np.round(deepcopy(outages) * int(10 ** precision))).astype(int)
adder = source4.index.min()*-1
source5 = deepcopy(source4).values
pass


def lightweight_v1():
    return lightweight_predict_future(source1, outages, ages, age_fractions, precision)


def lightweight_v2():
    return lightweight_v2_predict_future(source2, outages, ages, age_fractions, precision)


def lightweight_v3():
    return lightweight_v3_predict_future(source3, outages, ages, age_fractions, precision)


def lightweight_v4():
    return lightweight_predict_future_int(source4, outages4, ages4, age_fractions)

def lightweight_v5():
    return lightweight_predict_future_int_np(source5, outages4, ages4, age_fractions, adder)

def lightweight_v6():
    return lightweight_v2_predict_future_np(source5, outages4, ages4, age_fractions, adder)


if __name__ == '__main__':
    t1 = lightweight_v1().round(precision)
    t4 = lightweight_v4().round(precision)
    t5 = lightweight_v5().round(precision)
    t3 = lightweight_v3().round(precision)
    t2 = lightweight_v2().round(precision)
    t6 = lightweight_v6().round(precision)
    t4.index = (t4.index / 10 ** precision).round(precision)
    t1.index = t1.index.round(precision)
    t2.index = t2.index.round(precision)
    t3.index = t3.index.round(precision)
    pd.testing.assert_series_equal(t1, t2)
    pd.testing.assert_series_equal(t1, t3)
    pd.testing.assert_series_equal(t1, t4)
    assert np.allclose(t5, t6)
    assert np.allclose(t1.values, t5)
    assert np.allclose(t1.values, t6)

