import numpy as np

import spherely

import pytest


@pytest.mark.parametrize(
    "geog, expected",
    [
        # (spherely.Point(0, 0), spherely.Point(0, 0)),
        (
            spherely.LineString([(0, 0), (0, 2), (2, 2)]),
            spherely.Polygon([(0, 0), (0, 2), (2, 2)])
        ),
        # (
        #     spherely.Polygon([(0, 0), (0, 2), (2, 2), (1, 0.5)]),
        #     spherely.Polygon([(0, 0), (0, 2), (2, 2)])
        # ),
    ],
)
def test_convex_hull(geog, expected) -> None:
    # test array + scalar
    actual = spherely.convex_hull(geog)
    assert isinstance(actual, spherely.Geography)
    assert spherely.equals(actual, expected)

    actual = spherely.convex_hull([geog])
    assert isinstance(actual, np.ndarray)
    actual = actual[0]
    assert isinstance(actual, spherely.Geography)
    assert spherely.equals(actual, expected)
