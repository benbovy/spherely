import numpy as np

import s2shapely


def test_intersects():
    # test array + scalar
    a = np.array([s2shapely.LineString([(40, 8), (60, 8)]), s2shapely.LineString([(20, 0), (30, 0)])])
    b = s2shapely.LineString([(50, 5), (50, 10)])

    actual = s2shapely.intersects(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a = s2shapely.Point(50, 8)
    b = s2shapely.Point(20, 5)
    assert not s2shapely.intersects(a, b)


def test_equals():
    # test array + scalar
    a = np.array([s2shapely.LineString([(40, 8), (60, 8)]), s2shapely.LineString([(20, 0), (30, 0)])])
    b = s2shapely.Point(50, 8)

    actual = s2shapely.equals(a, b)
    expected = np.array([False, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a = s2shapely.Point(50, 8)
    b = s2shapely.Point(50, 8)
    assert s2shapely.equals(a, b)
