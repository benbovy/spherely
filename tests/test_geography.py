import pickle

import pytest
import numpy as np

import spherely


def test_geography_type() -> None:
    assert spherely.GeographyType.NONE.value == -1
    assert spherely.GeographyType.POINT.value == 0
    assert spherely.GeographyType.LINESTRING.value == 1
    assert spherely.GeographyType.POLYGON.value == 2
    assert spherely.GeographyType.MULTIPOINT.value == 3
    assert spherely.GeographyType.MULTILINESTRING.value == 4
    assert spherely.GeographyType.MULTIPOLYGON.value == 5
    assert spherely.GeographyType.GEOMETRYCOLLECTION.value == 6


def test_is_geography() -> None:
    arr = np.array([1, 2.33, spherely.create_point(30, 6)])

    actual = spherely.is_geography(arr)
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(actual, expected)


def test_not_geography_raise() -> None:
    arr = np.array([1, 2.33, spherely.create_point(30, 6)])

    with pytest.raises(TypeError, match="not a Geography object"):
        spherely.get_dimensions(arr)


def test_get_type_id() -> None:
    # array
    geog = np.array(
        [
            spherely.create_point(45, 50),
            spherely.create_multipoint([(5, 50), (6, 51)]),
            spherely.create_linestring([(5, 50), (6, 51)]),
            spherely.create_multilinestring([[(5, 50), (6, 51)], [(15, 60), (16, 61)]]),
            spherely.create_polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
            # with hole
            spherely.create_polygon(
                shell=[(5, 60), (6, 60), (6, 50), (5, 50)],
                holes=[[(5.1, 59), (5.9, 59), (5.9, 51), (5.1, 51)]],
            ),
            spherely.create_multipolygon(
                [
                    spherely.create_polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
                    spherely.create_polygon(
                        [(10, 100), (10, 160), (11, 160), (11, 100)]
                    ),
                ]
            ),
            spherely.create_collection([spherely.create_point(40, 50)]),
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
            spherely.GeographyType.GEOMETRYCOLLECTION.value,
        ]
    )
    np.testing.assert_array_equal(actual, expected)

    # scalar
    geog2 = spherely.create_point(45, 50)
    assert spherely.get_type_id(geog2) == spherely.GeographyType.POINT.value


def test_get_dimensions() -> None:
    # test n-d array
    expected = np.array([[0, 0], [1, 0]], dtype=np.int32)
    geog = np.array(
        [
            [spherely.create_point(5, 40), spherely.create_point(6, 30)],
            [
                spherely.create_linestring([(5, 50), (6, 51)]),
                spherely.create_point(4, 20),
            ],
        ]
    )
    actual = spherely.get_dimensions(geog)
    np.testing.assert_array_equal(actual, expected)

    # test scalar
    assert spherely.get_dimensions(spherely.create_point(5, 40)) == 0


def test_get_x_y() -> None:
    # scalar
    a = spherely.create_point(1.5, 2.6)
    assert spherely.get_x(a) == pytest.approx(1.5, abs=1e-14)
    assert spherely.get_y(a) == pytest.approx(2.6, abs=1e-14)

    # array
    arr = np.array(
        [
            spherely.create_point(0, 1),
            spherely.create_point(1, 2),
            spherely.create_point(2, 3),
        ]
    )

    actual = spherely.get_x(arr)
    expected = np.array([0, 1, 2], dtype="float64")
    np.testing.assert_allclose(actual, expected)

    actual = spherely.get_y(arr)
    expected = np.array([1, 2, 3], dtype="float64")
    np.testing.assert_allclose(actual, expected)

    # only points are supported
    with pytest.raises(ValueError):
        spherely.get_x(spherely.create_linestring([(0, 1), (1, 2)]))

    with pytest.raises(ValueError):
        spherely.get_y(spherely.create_linestring([(0, 1), (1, 2)]))


@pytest.mark.parametrize(
    "empty_geog, expected",
    [
        (spherely.create_point(), 0),
        (spherely.create_linestring(), 1),
        (spherely.create_polygon(), 2),
        (spherely.create_collection([]), -1),
    ],
)
def test_get_dimensions_empty(empty_geog, expected) -> None:
    assert spherely.get_dimensions(empty_geog) == expected


def test_get_dimensions_collection() -> None:
    geog = spherely.create_collection(
        [spherely.create_point(0, 0), spherely.create_polygon([(0, 0), (1, 1), (2, 0)])]
    )
    assert spherely.get_dimensions(geog) == 2


def test_prepare() -> None:
    # test array
    geog = np.array(
        [spherely.create_point(50, 45), spherely.create_linestring([(5, 50), (6, 51)])]
    )
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


def test_equality() -> None:
    p1 = spherely.create_point(1, 1)
    p2 = spherely.create_point(1, 1)
    p3 = spherely.create_point(2, 2)

    assert p1 == p1
    assert p1 == p2
    assert not p1 == p3

    line1 = spherely.create_linestring([(1, 1), (2, 2), (3, 3)])
    line2 = spherely.create_linestring([(3, 3), (2, 2), (1, 1)])

    assert line1 == line2

    poly1 = spherely.create_polygon([(1, 1), (3, 1), (2, 3)])
    poly2 = spherely.create_polygon([(2, 3), (1, 1), (3, 1)])
    poly3 = spherely.create_polygon([(2, 3), (3, 1), (1, 1)])

    assert p1 != poly1
    assert line1 != poly1
    assert poly1 == poly2
    assert poly2 == poly3
    assert poly1 == poly3

    coll1 = (spherely.create_collection([spherely.create_point(40, 50)]),)
    coll2 = (spherely.create_collection([spherely.create_point(40, 50)]),)

    assert coll1 == coll2


@pytest.mark.parametrize(
    "geog",
    [
        spherely.create_point(45, 50),
        spherely.create_multipoint([(5, 50), (6, 51)]),
        spherely.create_linestring([(5, 50), (6, 51)]),
        spherely.create_multilinestring([[(5, 50), (6, 51)], [(15, 60), (16, 61)]]),
        spherely.create_polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
        spherely.create_multipolygon(
            [
                spherely.create_polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
                spherely.create_polygon([(10, 100), (10, 160), (11, 160), (11, 100)]),
            ]
        ),
        spherely.create_collection([spherely.create_point(40, 50)]),
        # empty geography
        spherely.create_point(),
        spherely.create_linestring(),
        spherely.create_polygon(),
    ],
)
def test_pickle_roundtrip(geog):
    roundtripped = pickle.loads(pickle.dumps(geog))

    assert spherely.get_type_id(roundtripped) == spherely.get_type_id(geog)
    assert spherely.to_wkt(roundtripped) == spherely.to_wkt(geog)
    assert roundtripped == geog
