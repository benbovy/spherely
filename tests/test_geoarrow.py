import numpy as np
import pyarrow as pa
import geoarrow.pyarrow as ga

import pytest

import spherely


def test_from_geoarrow_wkt():

    arr = ga.as_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])

    result = spherely.from_geoarrow(arr)
    expected = spherely.create([1, 2, 3], [1, 2, 3])
    # object equality does not yet work
    # np.testing.assert_array_equal(result, expected)
    assert spherely.equals(result, expected).all()


def test_from_geoarrow_wkb():

    arr = ga.as_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    arr_wkb = ga.as_wkb(arr)

    result = spherely.from_geoarrow(arr_wkb)
    expected = spherely.create([1, 2, 3], [1, 2, 3])
    assert spherely.equals(result, expected).all()


def test_from_geoarrow_native():

    arr = ga.as_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    arr_point = ga.as_geoarrow(arr)

    result = spherely.from_geoarrow(arr_point)
    expected = spherely.create([1, 2, 3], [1, 2, 3])
    assert spherely.equals(result, expected).all()
