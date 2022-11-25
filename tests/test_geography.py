import pytest
import numpy as np

import sksphere


def test_point():
    point = sksphere.Point(40.2, 5.2)
    assert point.ndim == 0
    assert point.nshape == 1
    assert repr(point).startswith("POINT (5.2 40.")


def test_create():
    points = sksphere.create([40.0, 30.0], [5.0, 6.0])
    assert points.size == 2
    assert all([isinstance(p, sksphere.Point) for p in points])


@pytest.mark.parametrize(
    "points",
    [
        np.array([sksphere.Point(40, 5), sksphere.Point(30, 6)]),
        sksphere.create([40, 30], [5, 6]),
    ]
)
def test_nshape(points):
    expected = np.ones(2, dtype=np.int32)
    actual = sksphere.nshape(points)
    np.testing.assert_array_equal(actual, expected)
