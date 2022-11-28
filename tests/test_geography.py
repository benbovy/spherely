import pytest
import numpy as np

import s2shapely


def test_point():
    point = s2shapely.Point(40.2, 5.2)
    assert point.dimensions == 0
    assert point.nshape == 1
    assert repr(point).startswith("POINT (5.2 40.")


def test_create():
    points = s2shapely.create([40.0, 30.0], [5.0, 6.0])
    assert points.size == 2
    assert all([isinstance(p, s2shapely.Point) for p in points])


@pytest.mark.parametrize(
    "points",
    [
        np.array([s2shapely.Point(40, 5), s2shapely.Point(30, 6)]),
        s2shapely.create([40, 30], [5, 6]),
    ],
)
def test_nshape(points):
    expected = np.ones(2, dtype=np.int32)
    actual = s2shapely.nshape(points)
    np.testing.assert_array_equal(actual, expected)


def test_is_geography():
    arr = np.array([1, 2.33, s2shapely.Point(30, 6)])

    actual = s2shapely.is_geography(arr)
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(actual, expected)


def test_not_geography_raise():
    arr = np.array([1, 2.33, s2shapely.Point(30, 6)])

    with pytest.raises(TypeError, match="not a Geography object"):
        s2shapely.get_dimensions(arr)


def test_get_dimensions():
    # test n-d array
    expected = np.array([[0, 0], [1, 0]], dtype=np.int32)
    geog = np.array(
        [
            [s2shapely.Point(40, 5), s2shapely.Point(30, 6)],
            [s2shapely.LineString([(50, 5), (51, 6)]), s2shapely.Point(20, 4)],
        ]
    )
    actual = s2shapely.get_dimensions(geog)
    np.testing.assert_array_equal(actual, expected)

    # test scalar
    assert s2shapely.get_dimensions(s2shapely.Point(40, 5)) == 0


def test_prepare():
    # test array
    geog = np.array([s2shapely.Point(45, 50), s2shapely.LineString([(50, 5), (51, 6)])])
    np.testing.assert_array_equal(s2shapely.is_prepared(geog), np.array([False, False]))

    s2shapely.prepare(geog)
    np.testing.assert_array_equal(s2shapely.is_prepared(geog), np.array([True, True]))

    s2shapely.destroy_prepared(geog)
    np.testing.assert_array_equal(s2shapely.is_prepared(geog), np.array([False, False]))

    # test scalar
    geog = s2shapely.Point(45, 50)
    assert s2shapely.is_prepared(geog) is False

    s2shapely.prepare(geog)
    assert s2shapely.is_prepared(geog) is True

    s2shapely.destroy_prepared(geog)
    assert s2shapely.is_prepared(geog) is False
