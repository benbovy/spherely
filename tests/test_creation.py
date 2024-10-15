import pytest

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
        [spherely.point(5, 50), spherely.point(6, 51)],
    ],
)
def test_multipoint(points) -> None:
    multipoint = spherely.multipoint(points)
    assert multipoint.dimensions == 0
    assert multipoint.nshape == 1
    assert repr(multipoint).startswith("MULTIPOINT ((5 50)")


def test_multipoint_invalid_geography() -> None:
    point = spherely.point(5, 50)
    # all types other than points to test the error formatting
    multipoint = spherely.multipoint([(5, 50), (6, 61)])
    line = spherely.linestring([(5, 50), (6, 61)])
    multiline = spherely.multilinestring([line, line])
    polygon = spherely.polygon([(5, 50), (6, 61), (5, 61)])
    collection = spherely.geography_collection([point, line])

    with pytest.raises(
        TypeError, match=r"invalid Geography type \(expected POINT, found MULTIPOINT\)"
    ):
        spherely.multipoint([point, multipoint])
    with pytest.raises(
        TypeError, match=r"invalid Geography type \(expected POINT, found LINESTRING\)"
    ):
        spherely.multipoint([point, line])
    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found MULTILINESTRING\)",
    ):
        spherely.multipoint([point, multiline])
    with pytest.raises(
        TypeError, match=r"invalid Geography type \(expected POINT, found POLYGON\)"
    ):
        spherely.multipoint([point, polygon])
    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found GEOMETRYCOLLECTION\)",
    ):
        spherely.multipoint([point, collection])


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


def test_linestring_empty() -> None:
    line = spherely.linestring()
    assert repr(line).startswith("LINESTRING EMPTY")

    line = spherely.linestring(None)
    assert repr(line).startswith("LINESTRING EMPTY")

    line = spherely.linestring([])
    assert repr(line).startswith("LINESTRING EMPTY")

    with pytest.raises(ValueError, match="with empty component"):
        spherely.linestring([spherely.point(5, 50), spherely.point()])


def test_linestring_error() -> None:
    with pytest.raises(ValueError, match="at least 2 vertices"):
        spherely.linestring([(5, 50)])


def test_linestring_invalid_geography() -> None:
    point = spherely.point(5, 50)
    line = spherely.linestring([(5, 50), (6, 61)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found LINESTRING\)",
    ):
        spherely.linestring([point, line])


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


def test_multilinestring_invalid_geography() -> None:
    line = spherely.linestring([(5, 50), (6, 61)])
    polygon = spherely.polygon([(5, 50), (6, 61), (5, 61)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected LINESTRING, found POLYGON\)",
    ):
        spherely.multilinestring([line, polygon])


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

    point = spherely.point(1, 1)

    # CW and CCW polygons should be both valid
    assert spherely.contains(poly_ccw, point)
    assert spherely.contains(poly_cw, point)

    # CW polygon vertices reordered
    # TODO: better to test actual coordinate values when implemented
    assert repr(poly_cw) == "POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))"


def test_polygon_normalize_holes() -> None:
    poly_hole_ccw = spherely.polygon(
        shell=[(0, 0), (2, 0), (2, 2), (0, 2)],
        holes=[[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]],
    )

    # CCW polygon hole vertices reordered
    # TODO: better to test actual coordinate values when implemented
    assert (
        repr(poly_hole_ccw)
        == "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0), (0.5 1.5, 1.5 1.5, 1.5 0.5, 0.5 0.5, 0.5 1.5))"
    )


def test_polygon_empty() -> None:
    poly = spherely.polygon()
    assert repr(poly).startswith("POLYGON EMPTY")

    poly = spherely.polygon(None)
    assert repr(poly).startswith("POLYGON EMPTY")

    poly = spherely.polygon([])
    assert repr(poly).startswith("POLYGON EMPTY")

    with pytest.raises(ValueError, match="with empty component"):
        spherely.polygon(
            [spherely.point(5, 50), spherely.point(6, 50), spherely.point()]
        )


def test_polygon_invalid_geography() -> None:
    shell_points = [
        spherely.point(0, 0),
        spherely.point(2, 0),
        spherely.point(2, 2),
        spherely.point(0, 2),
    ]
    hole_points = [
        spherely.point(0.5, 0.5),
        spherely.point(1.5, 0.5),
        spherely.point(1.5, 1.5),
        spherely.point(0.5, 1.5),
    ]
    line = spherely.linestring([(3, 0), (3, 1)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found LINESTRING\)",
    ):
        spherely.polygon(shell_points + [line])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found LINESTRING\)",
    ):
        spherely.polygon(shell=shell_points, holes=[hole_points + [line]])


def test_multipolygons() -> None:
    poly1 = spherely.polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly2 = spherely.polygon(
        shell=[(4, 0), (6, 0), (6, 2), (4, 2)],
        holes=[[(4.5, 0.5), (5.5, 0.5), (5.5, 1.5), (4.5, 1.5)]],
    )
    print(spherely.GeographyType(spherely.get_type_id(poly2)))

    multipoly = spherely.multipolygon([poly1, poly2])
    print(multipoly)

    assert multipoly.dimensions == 2
    assert multipoly.nshape == 1
    assert repr(multipoly).startswith("MULTIPOLYGON (((0 0")


def test_multipolygon_invalid_geography() -> None:
    poly = spherely.polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    line = spherely.linestring([(3, 0), (3, 1)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POLYGON, found LINESTRING\)",
    ):
        spherely.multipolygon([poly, line])


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

    # TODO: test objects are copied (if/when we can update them in place)
    # (for now only test that original objects are preserved)
    assert [o.nshape for o in objs] == [1, 1, 1]

    # test nested collection
    # coll2 = spherely.geography_collection(objs + [coll])

    # assert repr(coll2).count("POINT") == 2
    # assert repr(coll2).count("LINESTRING") == 2
    # assert repr(coll2).count("POLYGON") == 2
    # assert repr(coll2).count("GEOMETRYCOLLECTION") == 2
