import numpy as np
import pytest

import spherely


@pytest.mark.parametrize(
    "geog, expected",
    [
        (spherely.point(0, 0), spherely.point(0, 0)),
        (
            spherely.linestring([(0, 0), (2, 0)]),
            spherely.point(1, 0),
        ),
        (
            spherely.polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
            spherely.point(1, 1),
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
        (spherely.point(0, 0), "GEOMETRYCOLLECTION EMPTY"),
        (spherely.linestring([(0, 0), (2, 0), (2, 2)]), "MULTIPOINT ((0 0), (2 2))"),
        (
            spherely.polygon([(0, 0), (0, 2), (2, 2), (0.5, 1.5)]),
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
            spherely.linestring([(0, 0), (2, 0), (2, 2)]),
            spherely.polygon([(0, 0), (2, 0), (2, 2)]),
        ),
        (
            spherely.polygon([(0, 0), (0, 2), (2, 2), (0.5, 1.5)]),
            spherely.polygon([(0, 0), (0, 2), (2, 2)]),
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
            spherely.point(0, 0),
            spherely.point(0, 90),
            np.pi / 2 * spherely.EARTH_RADIUS_METERS,
        ),
        (
            spherely.point(0, 90),
            spherely.point(90, 30),
            np.pi / 3 * spherely.EARTH_RADIUS_METERS,
        ),
        (
            spherely.polygon([(0, 0), (30, 60), (60, -30)]),
            spherely.point(0, 90),
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
        spherely.point(0, 90),
        spherely.point(0, 0),
        radius=1,
    )
    assert isinstance(actual, float)
    assert actual == pytest.approx(np.pi / 2)
