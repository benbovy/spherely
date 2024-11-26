import numpy as np
import pytest

import spherely


def test_get_x_y() -> None:
    # scalar
    a = spherely.point(1.5, 2.6)
    assert spherely.get_x(a) == pytest.approx(1.5, abs=1e-14)
    assert spherely.get_y(a) == pytest.approx(2.6, abs=1e-14)

    # array
    arr = np.array([spherely.point(0, 1), spherely.point(1, 2), spherely.point(2, 3)])

    actual = spherely.get_x(arr)
    expected = np.array([0, 1, 2], dtype="float64")
    np.testing.assert_allclose(actual, expected)

    actual = spherely.get_y(arr)
    expected = np.array([1, 2, 3], dtype="float64")
    np.testing.assert_allclose(actual, expected)

    # only points are supported
    with pytest.raises(ValueError):
        spherely.get_x(spherely.linestring([(0, 1), (1, 2)]))

    with pytest.raises(ValueError):
        spherely.get_y(spherely.linestring([(0, 1), (1, 2)]))
