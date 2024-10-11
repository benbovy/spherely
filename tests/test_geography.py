import pytest
import numpy as np

import spherely


def test_point() -> None:
    point = spherely.point(40.2, 5.2)
    assert point.dimensions == 0
    assert point.nshape == 1
    assert repr(point).startswith("POINT (40.2 5.2")


@pytest.mark.parametrize(
    "points",
    [
        [(5, 50), (6, 51)],
        [spherely.points(5, 50), spherely.points(6, 51)],
    ],
)
def test_multipoint(points) -> None:
    multipoint = spherely.multipoint(points)
    assert multipoint.dimensions == 0
    assert multipoint.nshape == 1
    assert repr(multipoint).startswith("MULTIPOINT ((5 50)")


@pytest.mark.parametrize(
    "points",
    [
        [(5, 50), (6, 51)],
        [spherely.point(5, 50), spherely.point(6, 51)],
    ],
)
def test_linestring(points) -> None:
    line = spherely.linestring(points)
    assert line.dimensions == 1
    assert line.nshape == 1
    assert repr(line).startswith("LINESTRING (5 50")


@pytest.mark.parametrize(
    "lines",
    [
        [[(5, 50), (6, 51)], [(15, 60), (16, 61)]],
        [
            [spherely.point(5, 50), spherely.point(6, 51)],
            [spherely.point(15, 60), spherely.point(16, 61)],
        ],
        [
            spherely.linestring([(5, 50), (6, 51)]),
            spherely.linestring([(15, 60), (16, 61)]),
        ],
    ],
)
def test_multilinestring(lines) -> None:
    multiline = spherely.multilinestring(lines)
    assert multiline.dimensions == 1
    assert multiline.nshape == 2
    assert repr(multiline).startswith("MULTILINESTRING ((5 50")


@pytest.mark.parametrize(
    "coords",
    [
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        [
            spherely.point(0, 0),
            spherely.point(2, 0),
            spherely.point(2, 2),
            spherely.point(0, 2),
        ],
    ],
)
def test_polygon(coords) -> None:
    poly = spherely.polygon(coords)
    assert poly.dimensions == 2
    assert poly.nshape == 1
    assert repr(poly).startswith("POLYGON ((0 0")


@pytest.mark.parametrize(
    "coords",
    [
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
    ],
)
def test_polygon_closing(coords) -> None:
    # support both manual and automated closing
    ring = spherely.polygon(coords)
    assert repr(ring).startswith("POLYGON ((0 0")
    assert repr(ring).endswith("0 0))")


def test_polygon_error() -> None:
    with pytest.raises(ValueError, match="polygon is not valid.*duplicate vertex.*"):
        spherely.polygon([(0, 0), (0, 2), (0, 2), (2, 0)])

    with pytest.raises(ValueError, match="polygon is not valid.*at least 3 vertices.*"):
        spherely.polygon([(0, 0), (0, 2)])

    with pytest.raises(ValueError, match="polygon is not valid.*Edge.*crosses.*"):
        spherely.polygon([(0, 0), (2, 0), (1, 2), (1, -2)])

    with pytest.raises(ValueError, match="polygon is not valid.*crosses.*"):
        # shell/hole rings are crossing each other
        spherely.polygon(
            shell=[(0, 0), (0, 4), (4, 4), (4, 0)],
            holes=[[(0, 1), (0, 5), (5, 5), (5, 1)]],
        )


def test_polygon_normalize() -> None:
    poly_ccw = spherely.polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly_cw = spherely.polygon([(0, 0), (0, 2), (2, 2), (2, 0)])

    point = spherely.points(1, 1)

    # CW and CCW polygons should be both valid
    assert spherely.contains(poly_ccw, point)
    assert spherely.contains(poly_cw, point)

    # CW polygon vertices reordered
    # TODO: better to test actual coordinate values when implemented
    assert repr(poly_cw) == "POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))"


def test_collection() -> None:
    objs = [
        spherely.point(0, 0),
        spherely.linestring([(0, 0), (1, 1)]),
        spherely.polygon([(0, 0), (1, 0), (1, 1)]),
    ]

    coll = spherely.geography_collection(objs)

    assert coll.dimensions == -1
    assert coll.nshape == 3
    assert repr(coll).startswith("GEOMETRYCOLLECTION (")

    # TODO: more robust test than using the WKT repr
    assert repr(coll).count("POINT") == 1
    assert repr(coll).count("LINESTRING") == 1
    assert repr(coll).count("POLYGON") == 1

    # TODO: test objects are copied
    # (for now only test that original objects are preserved)
    assert [o.nshape for o in objs] == [1, 1, 1]

    # test nested collection
    # coll2 = spherely.geography_collection(objs + [coll])

    # assert repr(coll2).count("POINT") == 2
    # assert repr(coll2).count("LINESTRING") == 2
    # assert repr(coll2).count("POLYGON") == 2
    # assert repr(coll2).count("GEOMETRYCOLLECTION") == 2


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
