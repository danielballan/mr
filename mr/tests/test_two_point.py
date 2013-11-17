from __future__ import division
import mr
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_almost_equal, assert_produces_warning)

def random_walk(N):
        return np.cumsum(np.random.randn(N))

def conformity(df):
    "Organize toy data to look like real data."
    return df.set_index('frame', drop=False).sort(['frame', 'probe']). \
        astype('float64')

class TestTwoPoint(unittest.TestCase):

    def setUp(self):
        N = 4 
        Y = 1
        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N),
                      'frame': np.arange(N), 'probe': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'probe': np.ones(N - 1)})
        self.steppers = conformity(pd.concat([a, b]))
        c = DataFrame({'x': np.arange(3*N, 2*N + 1, -1), 'y': np.zeros(N - 1),
                      'frame': np.arange(1, N), 'probe': np.ones(N -1)})
        self.colliders = pd.concat([a, c])
        self.separation = (c.x - a.x).dropna().values

    def test_para_motion(self):
        N = 4
        R = np.ones(N - 2)
        lf = np.arange(len(R)) + 1

        # Parallel steppers have no motion along connecting line.
        actual = mr.tp_corr(self.steppers, lagframes=lf)
        expected = DataFrame({'R': [1., 1., 1.],
            'para': [0., 0., 0.], 'perp': [1., 1., 4.]}, index=[1, 1, 2])
        assert_frame_equal(actual, expected)

    def test_overlong_lagframe_warns(self):
        N = 4
        R = np.ones(N - 1)
        lf = np.arange(len(R)) + 1
        f = lambda: mr.tp_corr(self.steppers, lagframes=lf)
        assert_produces_warning(f)

    def test_perp_motion(self):
        N = 4
        R = self.separation[:-1]
        lf = np.arange(len(R)) + 1

        # Colliding steppers have ALL motion along connecting line.
        actual = mr.tp_corr(self.colliders, lagframes=lf, bins=False)
        expected = DataFrame({'R': [11., 9., 11.],
            'para': [-1., -1., -4.], 'perp': [0., 0., 0.]}, index=[1, 1, 2])
        assert_frame_equal(actual, expected)
