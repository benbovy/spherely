from packaging.version import Version

import numpy as np
import pyarrow as pa
import geoarrow.pyarrow as ga

import pytest

import spherely


pytestmark = pytest.mark.skipif(
    Version(spherely.__s2geography_version__) < Version("0.2.0"),
    reason="Needs s2geography >= 0.2.0",
)


def test_from_geoarrow_wkt():

    arr = ga.as_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])

    result = spherely.from_geoarrow(arr)
    expected = spherely.points([1, 2, 3], [1, 2, 3])
    # object equality does not yet work
    # np.testing.assert_array_equal(result, expected)
    assert spherely.equals(result, expected).all()

    # without extension type
    arr = pa.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    result = spherely.from_geoarrow(arr, geometry_encoding="WKT")
    assert spherely.equals(result, expected).all()


def test_from_geoarrow_wkb():

    arr = ga.as_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    arr_wkb = ga.as_wkb(arr)

    result = spherely.from_geoarrow(arr_wkb)
    expected = spherely.points([1, 2, 3], [1, 2, 3])
    assert spherely.equals(result, expected).all()

    # without extension type
    arr_wkb = ga.as_wkb(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    arr = arr_wkb.cast(pa.binary())
    result = spherely.from_geoarrow(arr, geometry_encoding="WKB")
    assert spherely.equals(result, expected).all()


def test_from_geoarrow_native():

    arr = ga.as_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    arr_point = ga.as_geoarrow(arr)

    result = spherely.from_geoarrow(arr_point)
    expected = spherely.points([1, 2, 3], [1, 2, 3])
    assert spherely.equals(result, expected).all()


polygon_with_bad_hole_wkt = (
    "POLYGON "
    "((20 35, 10 30, 10 10, 30 5, 45 20, 20 35),"
    "(30 20, 20 25, 20 15, 30 20))"
)


def test_from_geoarrow_oriented():
    # by default re-orients the inner ring
    arr = ga.as_geoarrow([polygon_with_bad_hole_wkt])

    result = spherely.from_geoarrow(arr)
    assert (
        str(result[0])
        == "POLYGON ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (20 15, 20 25, 30 20, 20 15))"
    )

    # if we force to not orient, we get an error
    with pytest.raises(RuntimeError, match="Inconsistent loop orientations detected"):
        spherely.from_geoarrow(arr, oriented=True)


def test_from_wkt_planar():
    arr = ga.as_geoarrow(["LINESTRING (-64 45, 0 45)"])
    result = spherely.from_geoarrow(arr)
    assert spherely.distance(result, spherely.point(-30.1, 45)) > 10000

    result = spherely.from_geoarrow(arr, planar=True)
    assert spherely.distance(result, spherely.point(-30.1, 45)) < 100

    result = spherely.from_geoarrow(arr, planar=True, tessellate_tolerance=10)
    assert spherely.distance(result, spherely.point(-30.1, 45)) < 10


def test_from_geoarrow_no_extension_type():
    arr = pa.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])

    with pytest.raises(RuntimeError, match="Expected extension type"):
        spherely.from_geoarrow(arr)


def test_from_geoarrow_invalid_encoding():
    arr = pa.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])

    with pytest.raises(ValueError, match="'geometry_encoding' should be one"):
        spherely.from_geoarrow(arr, geometry_encoding="point")


def test_from_geoarrow_no_arrow_object():
    with pytest.raises(ValueError, match="input should be an Arrow-compatible array"):
        spherely.from_geoarrow(np.array(["POINT (1 1)"], dtype=object))
