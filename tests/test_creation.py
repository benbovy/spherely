import pytest

import spherely


def test_point() -> None:
    point = spherely.create_point(40.2, 5.2)
    assert repr(point).startswith("POINT (40.2 5.2")


def test_point_empty() -> None:
    point = spherely.create_point()
    assert repr(point).startswith("POINT EMPTY")

    point = spherely.create_point(None, None)
    assert repr(point).startswith("POINT EMPTY")


@pytest.mark.parametrize(
    "points",
    [
        [(5, 50), (6, 51)],
        [spherely.create_point(5, 50), spherely.create_point(6, 51)],
    ],
)
def test_multipoint(points) -> None:
    multipoint = spherely.create_multipoint(points)
    assert repr(multipoint).startswith("MULTIPOINT ((5 50)")


def test_multipoint_invalid_geography() -> None:
    point = spherely.create_point(5, 50)
    # all types other than points to test the error formatting
    multipoint = spherely.create_multipoint([(5, 50), (6, 61)])
    line = spherely.create_linestring([(5, 50), (6, 61)])
    multiline = spherely.create_multilinestring([line, line])
    polygon = spherely.create_polygon([(5, 50), (6, 61), (5, 61)])
    collection = spherely.create_collection([point, line])

    with pytest.raises(
        TypeError, match=r"invalid Geography type \(expected POINT, found MULTIPOINT\)"
    ):
        spherely.create_multipoint([point, multipoint])
    with pytest.raises(
        TypeError, match=r"invalid Geography type \(expected POINT, found LINESTRING\)"
    ):
        spherely.create_multipoint([point, line])
    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found MULTILINESTRING\)",
    ):
        spherely.create_multipoint([point, multiline])
    with pytest.raises(
        TypeError, match=r"invalid Geography type \(expected POINT, found POLYGON\)"
    ):
        spherely.create_multipoint([point, polygon])
    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found GEOMETRYCOLLECTION\)",
    ):
        spherely.create_multipoint([point, collection])


@pytest.mark.parametrize(
    "points",
    [
        [(5, 50), (6, 51)],
        [spherely.create_point(5, 50), spherely.create_point(6, 51)],
    ],
)
def test_linestring(points) -> None:
    line = spherely.create_linestring(points)
    assert repr(line).startswith("LINESTRING (5 50")


def test_linestring_empty() -> None:
    line = spherely.create_linestring()
    assert repr(line).startswith("LINESTRING EMPTY")

    line = spherely.create_linestring(None)
    assert repr(line).startswith("LINESTRING EMPTY")

    line = spherely.create_linestring([])
    assert repr(line).startswith("LINESTRING EMPTY")

    with pytest.raises(ValueError, match="with empty component"):
        spherely.create_linestring(
            [spherely.create_point(5, 50), spherely.create_point()]
        )


def test_linestring_error() -> None:
    with pytest.raises(ValueError, match="at least 2 vertices"):
        spherely.create_linestring([(5, 50)])


def test_linestring_invalid_geography() -> None:
    point = spherely.create_point(5, 50)
    line = spherely.create_linestring([(5, 50), (6, 61)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found LINESTRING\)",
    ):
        spherely.create_linestring([point, line])


@pytest.mark.parametrize(
    "lines",
    [
        [[(5, 50), (6, 51)], [(15, 60), (16, 61)]],
        [
            [spherely.create_point(5, 50), spherely.create_point(6, 51)],
            [spherely.create_point(15, 60), spherely.create_point(16, 61)],
        ],
        [
            spherely.create_linestring([(5, 50), (6, 51)]),
            spherely.create_linestring([(15, 60), (16, 61)]),
        ],
    ],
)
def test_multilinestring(lines) -> None:
    multiline = spherely.create_multilinestring(lines)
    assert repr(multiline).startswith("MULTILINESTRING ((5 50")


def test_multilinestring_invalid_geography() -> None:
    line = spherely.create_linestring([(5, 50), (6, 61)])
    polygon = spherely.create_polygon([(5, 50), (6, 61), (5, 61)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected LINESTRING, found POLYGON\)",
    ):
        spherely.create_multilinestring([line, polygon])


@pytest.mark.parametrize(
    "coords",
    [
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        [
            spherely.create_point(0, 0),
            spherely.create_point(2, 0),
            spherely.create_point(2, 2),
            spherely.create_point(0, 2),
        ],
    ],
)
def test_polygon(coords) -> None:
    poly = spherely.create_polygon(coords)
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
    ring = spherely.create_polygon(coords)
    assert repr(ring).startswith("POLYGON ((0 0")
    assert repr(ring).endswith("0 0))")


def test_polygon_error() -> None:
    with pytest.raises(ValueError, match="polygon is not valid.*duplicate vertex.*"):
        spherely.create_polygon([(0, 0), (0, 2), (0, 2), (2, 0)])

    with pytest.raises(ValueError, match="polygon is not valid.*at least 3 vertices.*"):
        spherely.create_polygon([(0, 0), (0, 2)])

    with pytest.raises(ValueError, match="polygon is not valid.*Edge.*crosses.*"):
        spherely.create_polygon([(0, 0), (2, 0), (1, 2), (1, -2)])

    with pytest.raises(ValueError, match="polygon is not valid.*crosses.*"):
        # shell/hole rings are crossing each other
        spherely.create_polygon(
            shell=[(0, 0), (0, 4), (4, 4), (4, 0)],
            holes=[[(0, 1), (0, 5), (5, 5), (5, 1)]],
        )


def test_polygon_normalize() -> None:
    poly_ccw = spherely.create_polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly_cw = spherely.create_polygon([(0, 0), (0, 2), (2, 2), (2, 0)])

    point = spherely.create_point(1, 1)

    # CW and CCW polygons should be both valid
    assert spherely.contains(poly_ccw, point)
    assert spherely.contains(poly_cw, point)

    # CW polygon vertices reordered
    # TODO: better to test actual coordinate values when implemented
    assert repr(poly_cw) == "POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))"


@pytest.mark.parametrize(
    "shell",
    [
        [(0, 0), (0, 2), (2, 2), (2, 0)],
        [
            spherely.create_point(0, 0),
            spherely.create_point(0, 2),
            spherely.create_point(2, 2),
            spherely.create_point(2, 0),
        ],
    ],
)
def test_polygon_oriented(shell) -> None:
    # "CW" polygon + oriented=True => the polygon's interior is the largest
    # area of the sphere divided by the polygon's ring
    poly_cw = spherely.create_polygon(shell, oriented=True)

    point = spherely.create_point(1, 1)

    # point above is NOT in polygon's interior
    assert not spherely.contains(poly_cw, point)

    # polygon vertices not reordered
    # TODO: better to test actual coordinate values when implemented
    assert repr(poly_cw) == "POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"


@pytest.mark.parametrize(
    "shell,holes",
    [
        (
            [(0, 0), (2, 0), (2, 2), (0, 2)],
            [[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]],
        ),
        (
            [
                spherely.create_point(0, 0),
                spherely.create_point(2, 0),
                spherely.create_point(2, 2),
                spherely.create_point(0, 2),
            ],
            [
                [
                    spherely.create_point(0.5, 0.5),
                    spherely.create_point(1.5, 0.5),
                    spherely.create_point(1.5, 1.5),
                    spherely.create_point(0.5, 1.5),
                ]
            ],
        ),
    ],
)
def test_polygon_holes(shell, holes) -> None:
    poly = spherely.create_polygon(shell, holes=holes)
    assert repr(poly).startswith("POLYGON ((0 0")


def test_polygon_mixed_types_not_supported() -> None:
    shell = [
        spherely.create_point(0, 0),
        spherely.create_point(2, 0),
        spherely.create_point(2, 2),
        spherely.create_point(0, 2),
    ]
    holes = [[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]]

    with pytest.raises(TypeError, match="incompatible function arguments"):
        spherely.create_polygon(shell, holes=holes)  # type: ignore

    with pytest.raises(TypeError, match="incompatible function arguments"):
        spherely.create_polygon(None, holes=holes)  # type: ignore


def test_polygon_normalize_holes() -> None:
    poly_hole_ccw = spherely.create_polygon(
        shell=[(0, 0), (2, 0), (2, 2), (0, 2)],
        holes=[[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]],
    )

    # CCW polygon hole vertices reordered
    # TODO: better to test actual coordinate values when implemented
    assert (
        repr(poly_hole_ccw)
        == "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0), (0.5 1.5, 1.5 1.5, 1.5 0.5, 0.5 0.5, 0.5 1.5))"
    )


def test_polygon_oriented_holes() -> None:
    # CCW polygon hole vertices + oriented=True => error
    with pytest.raises(ValueError, match="Inconsistent loop orientations detected"):
        spherely.create_polygon(
            shell=[(0, 0), (2, 0), (2, 2), (0, 2)],
            holes=[[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]],
            oriented=True,
        )


def test_polygon_empty() -> None:
    poly = spherely.create_polygon()
    assert repr(poly).startswith("POLYGON EMPTY")

    poly = spherely.create_polygon(None)
    assert repr(poly).startswith("POLYGON EMPTY")

    poly = spherely.create_polygon([])
    assert repr(poly).startswith("POLYGON EMPTY")

    with pytest.raises(ValueError, match="with empty component"):
        spherely.create_polygon(
            [
                spherely.create_point(5, 50),
                spherely.create_point(6, 50),
                spherely.create_point(),
            ]
        )


def test_polygon_empty_shell_with_holes() -> None:
    holes = [[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]]

    with pytest.raises(ValueError, match="polygon shell is empty but found holes"):
        spherely.create_polygon([], holes=holes)


def test_polygon_invalid_geography() -> None:
    shell_points = [
        spherely.create_point(0, 0),
        spherely.create_point(2, 0),
        spherely.create_point(2, 2),
        spherely.create_point(0, 2),
    ]
    hole_points = [
        spherely.create_point(0.5, 0.5),
        spherely.create_point(1.5, 0.5),
        spherely.create_point(1.5, 1.5),
        spherely.create_point(0.5, 1.5),
    ]
    line = spherely.create_linestring([(3, 0), (3, 1)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found LINESTRING\)",
    ):
        spherely.create_polygon(shell_points + [line])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POINT, found LINESTRING\)",
    ):
        spherely.create_polygon(shell=shell_points, holes=[hole_points + [line]])


def test_multipolygons() -> None:
    poly1 = spherely.create_polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly2 = spherely.create_polygon(
        shell=[(4, 0), (6, 0), (6, 2), (4, 2)],
        holes=[[(4.5, 0.5), (5.5, 0.5), (5.5, 1.5), (4.5, 1.5)]],
    )

    multipoly = spherely.create_multipolygon([poly1, poly2])

    assert repr(multipoly).startswith("MULTIPOLYGON (((0 0")


def test_multipolygons_oriented() -> None:
    # same than `test_polygon_oriented`: make sure that it works for multipolygon too
    poly_cw = spherely.create_polygon([(0, 0), (0, 2), (2, 2), (2, 0)], oriented=True)

    # original polygon loops are cloned (so shouldn't be normalized) before being passed
    # to the multipolygon
    multipoly = spherely.create_multipolygon([poly_cw])

    point = spherely.create_point(1, 1)

    assert not spherely.contains(multipoly, point)


def test_multipolygon_invalid_geography() -> None:
    poly = spherely.create_polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    line = spherely.create_linestring([(3, 0), (3, 1)])

    with pytest.raises(
        TypeError,
        match=r"invalid Geography type \(expected POLYGON, found LINESTRING\)",
    ):
        spherely.create_multipolygon([poly, line])


def test_collection() -> None:
    objs = [
        spherely.create_point(0, 0),
        spherely.create_linestring([(0, 0), (1, 1)]),
        spherely.create_polygon([(0, 0), (1, 0), (1, 1)]),
    ]

    coll = spherely.create_collection(objs)

    assert repr(coll).startswith("GEOMETRYCOLLECTION (")

    # TODO: more robust test than using the WKT repr
    assert repr(coll).count("POINT") == 1
    assert repr(coll).count("LINESTRING") == 1
    assert repr(coll).count("POLYGON") == 1

    # test nested collection
    coll2 = spherely.create_collection(objs + [coll])

    assert repr(coll2).count("POINT") == 2
    assert repr(coll2).count("LINESTRING") == 2
    assert repr(coll2).count("POLYGON") == 2
    assert repr(coll2).count("GEOMETRYCOLLECTION") == 2
