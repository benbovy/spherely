import pytest
import numpy as np

import spherely


def test_is_geography() -> None:
    arr = np.array([1, 2.33, spherely.point(30, 6)])

    actual = spherely.is_geography(arr)
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(actual, expected)


def test_not_geography_raise() -> None:
    arr = np.array([1, 2.33, spherely.point(30, 6)])

    with pytest.raises(TypeError, match="not a Geography object"):
        spherely.get_dimensions(arr)


def test_get_type_id() -> None:
    # array
    geog = np.array(
        [
            spherely.point(45, 50),
            spherely.multipoint([(5, 50), (6, 51)]),
            spherely.linestring([(5, 50), (6, 51)]),
            spherely.multilinestring([[(5, 50), (6, 51)], [(15, 60), (16, 61)]]),
            spherely.polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
            # with hole
            spherely.polygon(
                shell=[(5, 60), (6, 60), (6, 50), (5, 50)],
                holes=[[(5.1, 59), (5.9, 59), (5.9, 51), (5.1, 51)]],
            ),
            spherely.multipolygon(
                [
                    spherely.polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
                    spherely.polygon([(10, 100), (10, 160), (11, 160), (11, 100)]),
                ]
            ),
            spherely.geography_collection([spherely.point(40, 50)]),
        ]
    )
    actual = spherely.get_type_id(geog)
    expected = np.array(
        [
            spherely.GeographyType.POINT.value,
            spherely.GeographyType.MULTIPOINT.value,
            spherely.GeographyType.LINESTRING.value,
            spherely.GeographyType.MULTILINESTRING.value,
            spherely.GeographyType.POLYGON.value,
            spherely.GeographyType.POLYGON.value,
            spherely.GeographyType.MULTIPOLYGON.value,
            spherely.GeographyType.GEOGRAPHYCOLLECTION.value,
        ]
    )
    np.testing.assert_array_equal(actual, expected)

    # scalar
    geog2 = spherely.point(45, 50)
    assert spherely.get_type_id(geog2) == spherely.GeographyType.POINT.value


def test_get_dimensions() -> None:
    # test n-d array
    expected = np.array([[0, 0], [1, 0]], dtype=np.int32)
    geog = np.array(
        [
            [spherely.point(5, 40), spherely.point(6, 30)],
            [spherely.linestring([(5, 50), (6, 51)]), spherely.point(4, 20)],
        ]
    )
    actual = spherely.get_dimensions(geog)
    np.testing.assert_array_equal(actual, expected)

    # test scalar
    assert spherely.get_dimensions(spherely.point(5, 40)) == 0


def test_prepare() -> None:
    # test array
    geog = np.array([spherely.point(50, 45), spherely.linestring([(5, 50), (6, 51)])])
    np.testing.assert_array_equal(spherely.is_prepared(geog), np.array([False, False]))

    spherely.prepare(geog)
    np.testing.assert_array_equal(spherely.is_prepared(geog), np.array([True, True]))

    spherely.destroy_prepared(geog)
    np.testing.assert_array_equal(spherely.is_prepared(geog), np.array([False, False]))

    # test scalar
    geog2 = spherely.points(45, 50)
    assert spherely.is_prepared(geog2) is False

    spherely.prepare(geog2)
    assert spherely.is_prepared(geog2) is True

    spherely.destroy_prepared(geog2)
    assert spherely.is_prepared(geog2) is False
