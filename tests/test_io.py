import numpy as np
import pytest

import spherely


def test_from_wkt():
    result = spherely.from_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    expected = spherely.create([1, 2, 3], [1, 2, 3])
    # object equality does not yet work
    # np.testing.assert_array_equal(result, expected)
    assert spherely.equals(result, expected).all()

    # from explicit object dtype
    result = spherely.from_wkt(
        np.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"], dtype=object)
    )
    assert spherely.equals(result, expected).all()

    # from numpy string dtype
    result = spherely.from_wkt(
        np.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"], dtype="U")
    )
    assert spherely.equals(result, expected).all()


def test_from_wkt_invalid():
    # TODO can we provide better error type?
    with pytest.raises(RuntimeError):
        spherely.from_wkt(["POINT (1)"])


def test_from_wkt_wrong_type():
    with pytest.raises(TypeError, match="expected bytes, int found"):
        spherely.from_wkt([1])

    # TODO support missing values
    with pytest.raises(TypeError, match="expected bytes, NoneType found"):
        spherely.from_wkt(["POINT (1 1)", None])


def test_to_wkt():
    arr = spherely.create([1.1, 2, 3], [1.1, 2, 3])
    result = spherely.to_wkt(arr)
    expected = np.array(["POINT (1.1 1.1)", "POINT (2 2)", "POINT (3 3)"], dtype=object)
    np.testing.assert_array_equal(result, expected)
