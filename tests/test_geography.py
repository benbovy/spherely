import pytest
import numpy as np

import spherely


def test_point() -> None:
    point = spherely.Point(40.2, 5.2)
    assert point.dimensions == 0
    assert point.nshape == 1
    assert repr(point).startswith("POINT (5.2 40.")


@pytest.mark.parametrize(
    "coords",
    [
        [(50, 5), (51, 6)],
        [spherely.Point(50, 5), spherely.Point(51, 6)],
    ],
)
def test_linestring(coords) -> None:
    line = spherely.LineString(coords)
    assert line.dimensions == 1
    assert line.nshape == 1
    assert repr(line).startswith("LINESTRING (5 50")


def test_is_geography() -> None:
    arr = np.array([1, 2.33, spherely.Point(30, 6)])

    actual = spherely.is_geography(arr)
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(actual, expected)


def test_not_geography_raise() -> None:
    arr = np.array([1, 2.33, spherely.Point(30, 6)])

    with pytest.raises(TypeError, match="not a Geography object"):
        spherely.get_dimensions(arr)


def test_get_type_id() -> None:
    # array
    geog = np.array([spherely.Point(45, 50), spherely.LineString([(50, 5), (51, 6)])])
    actual = spherely.get_type_id(geog)
    expected = np.array(
        [spherely.GeographyType.POINT.value, spherely.GeographyType.LINESTRING.value]
    )
    np.testing.assert_array_equal(actual, expected)

    # scalar
    geog2 = spherely.Point(45, 50)
    assert spherely.get_type_id(geog2) == spherely.GeographyType.POINT.value


def test_get_dimensions() -> None:
    # test n-d array
    expected = np.array([[0, 0], [1, 0]], dtype=np.int32)
    geog = np.array(
        [
            [spherely.Point(40, 5), spherely.Point(30, 6)],
            [spherely.LineString([(50, 5), (51, 6)]), spherely.Point(20, 4)],
        ]
    )
    actual = spherely.get_dimensions(geog)
    np.testing.assert_array_equal(actual, expected)

    # test scalar
    assert spherely.get_dimensions(spherely.Point(40, 5)) == 0


def test_prepare() -> None:
    # test array
    geog = np.array([spherely.Point(45, 50), spherely.LineString([(50, 5), (51, 6)])])
    np.testing.assert_array_equal(spherely.is_prepared(geog), np.array([False, False]))

    spherely.prepare(geog)
    np.testing.assert_array_equal(spherely.is_prepared(geog), np.array([True, True]))

    spherely.destroy_prepared(geog)
    np.testing.assert_array_equal(spherely.is_prepared(geog), np.array([False, False]))

    # test scalar
    geog2 = spherely.Point(45, 50)
    assert spherely.is_prepared(geog2) is False

    spherely.prepare(geog2)
    assert spherely.is_prepared(geog2) is True

    spherely.destroy_prepared(geog2)
    assert spherely.is_prepared(geog2) is False


def test_create() -> None:
    points = spherely.create([40.0, 30.0], [5.0, 6.0])
    assert points.size == 2
    assert all([isinstance(p, spherely.Point) for p in points])


@pytest.mark.parametrize(
    "points",
    [
        np.array([spherely.Point(40, 5), spherely.Point(30, 6)]),
        spherely.create([40, 30], [5, 6]),
    ],
)
def test_nshape(points) -> None:
    expected = np.ones(2, dtype=np.int32)
    actual = spherely.nshape(points)
    np.testing.assert_array_equal(actual, expected)
