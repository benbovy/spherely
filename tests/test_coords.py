import numpy as np
import pytest

import spherely


def test_get_lat_lng() -> None:
    # scalar
    a = spherely.Point(1.5, 2.6)
    assert spherely.get_lat(a) == pytest.approx(1.5, abs=1e-14)
    assert spherely.get_lng(a) == pytest.approx(2.6, abs=1e-14)

    # array
    arr = np.array([spherely.Point(0, 1), spherely.Point(1, 2), spherely.Point(2, 3)])

    actual = spherely.get_lat(arr)
    expected = np.array([0, 1, 2], dtype="float64")
    np.testing.assert_allclose(actual, expected)

    actual = spherely.get_lng(arr)
    expected = np.array([1, 2, 3], dtype="float64")
    np.testing.assert_allclose(actual, expected)

    # only points are supported
    with pytest.raises(ValueError):
        spherely.get_lat(spherely.LineString([(0, 1), (1, 2)]))

    with pytest.raises(ValueError):
        spherely.get_lng(spherely.LineString([(0, 1), (1, 2)]))
