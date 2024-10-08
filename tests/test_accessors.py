import numpy as np
import pytest

import spherely


@pytest.mark.parametrize(
    "geog, expected",
    [
        (spherely.Point(0, 0), spherely.Point(0, 0)),
        (
            spherely.LineString([(0, 0), (0, 2)]),
            spherely.Point(0, 1),
        ),
        (
            spherely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            spherely.Point(1, 1),
        ),
    ],
)
def test_centroid(geog, expected) -> None:
    # scalar
    actual = spherely.centroid(geog)
    assert isinstance(actual, spherely.Point)
    # TODO add some way of testing almost equality
    # assert spherely.equals(actual, expected)

    # array
    actual = spherely.centroid([geog])
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert isinstance(actual, spherely.Point)
    # assert spherely.equals(actual, expected)


@pytest.mark.parametrize(
    "geog, expected",
    [
        (spherely.Point(0, 0), "GEOMETRYCOLLECTION EMPTY"),
        (spherely.LineString([(0, 0), (0, 2), (2, 2)]), "MULTIPOINT ((0 0), (2 2))"),
        (
            spherely.Polygon([(0, 0), (2, 0), (2, 2), (1.5, 0.5)]),
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
            spherely.LineString([(0, 0), (0, 2), (2, 2)]),
            spherely.Polygon([(0, 0), (0, 2), (2, 2)]),
        ),
        (
            spherely.Polygon([(0, 0), (2, 0), (2, 2), (1.5, 0.5)]),
            spherely.Polygon([(0, 0), (2, 0), (2, 2)]),
        ),
    ],
)
def test_convex_hull(geog, expected) -> None:
    # scalar
    actual = spherely.convex_hull(geog)
    assert isinstance(actual, spherely.Polygon)
    assert spherely.equals(actual, expected)

    # array
    actual = spherely.convex_hull([geog])
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert isinstance(actual, spherely.Polygon)
    assert spherely.equals(actual, expected)


@pytest.mark.parametrize(
    "geog_a, geog_b, expected",
    [
        (
            spherely.Point(0, 0),
            spherely.Point(90, 0),
            np.pi / 2 * spherely.EARTH_RADIUS_METERS,
        ),
        (
            spherely.Point(90, 0),
            spherely.Point(30, 90),
            np.pi / 3 * spherely.EARTH_RADIUS_METERS,
        ),
        (
            spherely.Polygon([(0, 0), (60, 30), (-30, 60)]),
            spherely.Point(90, 0),
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
        spherely.Point(90, 0),
        spherely.Point(0, 0),
        radius=1,
    )
    assert isinstance(actual, float)
    assert actual == pytest.approx(np.pi / 2)
