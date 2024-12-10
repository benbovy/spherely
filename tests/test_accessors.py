import math

import numpy as np
import pytest

import spherely


@pytest.mark.parametrize(
    "geog, expected",
    [
        (spherely.create_point(0, 0), spherely.create_point(0, 0)),
        (
            spherely.create_linestring([(0, 0), (2, 0)]),
            spherely.create_point(1, 0),
        ),
        (
            spherely.create_polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
            spherely.create_point(1, 1),
        ),
    ],
)
def test_centroid(geog, expected) -> None:
    # scalar
    actual = spherely.centroid(geog)
    assert spherely.get_type_id(actual) == spherely.GeographyType.POINT.value
    # TODO add some way of testing almost equality
    # assert spherely.equals(actual, expected)

    # array
    actual = spherely.centroid([geog])
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert spherely.get_type_id(actual) == spherely.GeographyType.POINT.value
    # assert spherely.equals(actual, expected)


@pytest.mark.parametrize(
    "geog, expected",
    [
        (spherely.create_point(0, 0), "GEOMETRYCOLLECTION EMPTY"),
        (
            spherely.create_linestring([(0, 0), (2, 0), (2, 2)]),
            "MULTIPOINT ((0 0), (2 2))",
        ),
        (
            spherely.create_polygon([(0, 0), (0, 2), (2, 2), (0.5, 1.5)]),
            "LINESTRING (0.5 1.5, 2 2, 0 2, 0 0, 0.5 1.5)",
        ),
    ],
)
def test_boundary(geog, expected) -> None:
    # scalar
    actual = spherely.boundary(geog)
    assert str(actual) == expected

    # array
    actual = spherely.boundary([geog])
    assert isinstance(actual, np.ndarray)
    assert str(actual[0]) == expected


@pytest.mark.parametrize(
    "geog, expected",
    [
        (
            spherely.create_linestring([(0, 0), (2, 0), (2, 2)]),
            spherely.create_polygon([(0, 0), (2, 0), (2, 2)]),
        ),
        (
            spherely.create_polygon([(0, 0), (0, 2), (2, 2), (0.5, 1.5)]),
            spherely.create_polygon([(0, 0), (0, 2), (2, 2)]),
        ),
    ],
)
def test_convex_hull(geog, expected) -> None:
    # scalar
    actual = spherely.convex_hull(geog)
    assert spherely.get_type_id(actual) == spherely.GeographyType.POLYGON.value
    assert spherely.equals(actual, expected)

    # array
    actual = spherely.convex_hull([geog])
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert spherely.get_type_id(actual) == spherely.GeographyType.POLYGON.value
    assert spherely.equals(actual, expected)


@pytest.mark.parametrize(
    "geog_a, geog_b, expected",
    [
        (
            spherely.create_point(0, 0),
            spherely.create_point(0, 90),
            np.pi / 2 * spherely.EARTH_RADIUS_METERS,
        ),
        (
            spherely.create_point(0, 90),
            spherely.create_point(90, 30),
            np.pi / 3 * spherely.EARTH_RADIUS_METERS,
        ),
        (
            spherely.create_polygon([(0, 0), (30, 60), (60, -30)]),
            spherely.create_point(0, 90),
            np.pi / 6 * spherely.EARTH_RADIUS_METERS,
        ),
    ],
)
def test_distance(geog_a, geog_b, expected) -> None:
    # scalar
    actual = spherely.distance(geog_a, geog_b)
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, 1e-9)

    # array
    actual = spherely.distance([geog_a], [geog_b])
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, 1e-9)


def test_distance_with_custom_radius() -> None:
    actual = spherely.distance(
        spherely.create_point(0, 90),
        spherely.create_point(0, 0),
        radius=1,
    )
    assert isinstance(actual, float)
    assert actual == pytest.approx(np.pi / 2)


def test_area() -> None:
    # scalar
    geog = spherely.create_polygon([(0, 0), (90, 0), (0, 90), (0, 0)])
    result = spherely.area(geog, radius=1)
    assert isinstance(result, float)
    expected = 4 * math.pi / 8
    assert result == pytest.approx(expected, 1e-9)

    result = spherely.area(geog)
    assert result == pytest.approx(expected * spherely.EARTH_RADIUS_METERS**2, 1e-9)

    # array
    actual = spherely.area([geog], radius=1)
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert isinstance(actual, float)
    assert actual == pytest.approx(4 * math.pi / 8, 1e-9)


@pytest.mark.parametrize(
    "geog",
    [
        "POINT (-64 45)",
        "POINT EMPTY",
        "LINESTRING (0 0, 1 1)",
        "LINESTRING EMPTY",
        "POLYGON EMPTY",
    ],
)
def test_area_empty(geog) -> None:
    assert spherely.area(spherely.from_wkt(geog)) == 0


def test_length() -> None:
    geog = spherely.create_linestring([(0, 0), (1, 0)])
    result = spherely.length(geog, radius=1)
    assert isinstance(result, float)
    expected = 1.0 * np.pi / 180.0
    assert result == pytest.approx(expected, 1e-9)

    actual = spherely.length([geog], radius=1)
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, 1e-9)


@pytest.mark.parametrize(
    "geog",
    [
        "POINT (0 0)",
        "POINT EMPTY",
        "POLYGON EMPTY",
        "POLYGON ((0 0, 0 1, 1 0, 0 0))",
    ],
)
def test_length_invalid(geog) -> None:
    assert spherely.length(spherely.from_wkt(geog)) == 0.0


def test_perimeter() -> None:
    geog = spherely.create_polygon([(0, 0), (0, 90), (90, 90), (90, 0), (0, 0)])
    result = spherely.perimeter(geog, radius=1)
    assert isinstance(result, float)
    expected = 3 * 90 * np.pi / 180.0
    assert result == pytest.approx(expected, 1e-9)

    actual = spherely.perimeter([geog], radius=1)
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, 1e-9)


@pytest.mark.parametrize(
    "geog", ["POINT (0 0)", "POINT EMPTY", "LINESTRING (0 0, 1 0)", "POLYGON EMPTY"]
)
def test_perimeter_invalid(geog) -> None:
    assert spherely.perimeter(spherely.from_wkt(geog)) == 0.0
