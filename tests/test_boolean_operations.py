from packaging.version import Version

import pytest

import spherely

poly1 = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
poly2 = spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))")


@pytest.mark.parametrize(
    "geog1, geog2, expected",
    [
        ("POINT (30 10)", "POINT EMPTY", "POINT (30 10)"),
        ("POINT EMPTY", "POINT EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        (
            "LINESTRING (-45 0, 0 0)",
            "LINESTRING (0 0, 0 10)",
            "LINESTRING (-45 0, 0 0, 0 10)",
        ),
    ],
)
def test_union(geog1, geog2, expected) -> None:
    result = spherely.union(spherely.from_wkt(geog1), spherely.from_wkt(geog2))
    assert str(result) == expected


def test_union_polygon():
    result = spherely.union(poly1, poly2)

    expected_near = (
        spherely.area(spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"))
        + spherely.area(spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))"))
        - spherely.area(spherely.from_wkt("POLYGON ((5 5, 10 5, 10 15, 5 10, 5 5))"))
    )
    pytest.approx(spherely.area(result), expected_near)


@pytest.mark.parametrize(
    "geog1, geog2, expected",
    [
        ("POINT (30 10)", "POINT (30 10)", "POINT (30 10)"),
        (
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "LINESTRING (0 5, 10 5)",
            "LINESTRING (0 5, 10 5)",
        ),
    ],
)
def test_intersection(geog1, geog2, expected) -> None:
    result = spherely.intersection(spherely.from_wkt(geog1), spherely.from_wkt(geog2))
    assert str(result) == expected


def test_intersection_empty() -> None:
    result = spherely.intersection(poly1, spherely.from_wkt("POLYGON EMPTY"))
    # assert spherely.is_empty(result)
    assert str(result) == "GEOMETRYCOLLECTION EMPTY"

    result = spherely.intersection(spherely.from_wkt("POLYGON EMPTY"), poly1)
    assert str(result) == "GEOMETRYCOLLECTION EMPTY"

    result = spherely.intersection(
        spherely.from_wkt("POINT (0 1)"), spherely.from_wkt("POINT (1 2)")
    )
    assert str(result) == "GEOMETRYCOLLECTION EMPTY"


def test_intersection_lines() -> None:
    result = spherely.intersection(
        spherely.from_wkt("LINESTRING (-45 0, 45 0)"),
        spherely.from_wkt("LINESTRING (0 -10, 0 10)"),
    )
    assert str(result) == "POINT (0 0)"
    assert spherely.distance(result, spherely.from_wkt("POINT (0 0)")) == 0


def test_intersection_polygons() -> None:
    result = spherely.intersection(poly1, poly2)
    # TODO precision could be higher with snap level
    precision = 2 if Version(spherely.__s2geography_version__) < Version("0.2.0") else 1
    assert (
        spherely.to_wkt(result, precision=precision)
        == "POLYGON ((5 5, 10 5, 10 10, 5 10, 5 5))"
    )


def test_intersection_polygon_model() -> None:
    poly = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
    point = spherely.from_wkt("POINT (0 0)")

    result = spherely.intersection(poly, point)
    assert str(result) == "GEOMETRYCOLLECTION EMPTY"

    # TODO this will be different depending on the model


@pytest.mark.parametrize(
    "geog1, geog2, expected",
    [
        ("POINT (30 10)", "POINT EMPTY", "POINT (30 10)"),
        ("POINT EMPTY", "POINT EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        (
            "LINESTRING (0 0, 45 0)",
            "LINESTRING (0 0, 45 0)",
            "GEOMETRYCOLLECTION EMPTY",
        ),
    ],
)
def test_difference(geog1, geog2, expected) -> None:
    result = spherely.difference(spherely.from_wkt(geog1), spherely.from_wkt(geog2))
    assert spherely.equals(result, spherely.from_wkt(expected))


def test_difference_polygons() -> None:
    result = spherely.difference(poly1, poly2)
    expected_near = spherely.area(poly1) - spherely.area(
        spherely.from_wkt("POLYGON ((5 5, 10 5, 10 10, 5 10, 5 5))")
    )
    pytest.approx(spherely.area(result), expected_near)


@pytest.mark.parametrize(
    "geog1, geog2, expected",
    [
        ("POINT (30 10)", "POINT EMPTY", "POINT (30 10)"),
        ("POINT (30 10)", "POINT (30 10)", "GEOMETRYCOLLECTION EMPTY"),
        ("POINT (30 10)", "POINT (30 20)", "MULTIPOINT ((30 20), (30 10))"),
        (
            "LINESTRING (0 0, 45 0)",
            "LINESTRING (0 0, 45 0)",
            "GEOMETRYCOLLECTION EMPTY",
        ),
    ],
)
def test_symmetric_difference(geog1, geog2, expected) -> None:
    result = spherely.symmetric_difference(
        spherely.from_wkt(geog1), spherely.from_wkt(geog2)
    )
    assert spherely.equals(result, spherely.from_wkt(expected))


def test_symmetric_difference_polygons() -> None:
    result = spherely.symmetric_difference(poly1, poly2)
    expected_near = 2 * (
        spherely.area(poly1)
        - spherely.area(spherely.from_wkt("POLYGON ((5 5, 10 5, 10 10, 5 10, 5 5))"))
    )
    pytest.approx(spherely.area(result), expected_near)
