from packaging.version import Version

import numpy as np
<<<<<<< HEAD
=======
import pyarrow as pa
import geoarrow.pyarrow as ga
>>>>>>> e182953 (ENH: export to geoarrow output)

import pytest

import spherely


pytestmark = pytest.mark.skipif(
    Version(spherely.__s2geography_version__) < Version("0.2.0"),
    reason="Needs s2geography >= 0.2.0",
)

pa = pytest.importorskip("pyarrow")
ga = pytest.importorskip("geoarrow.pyarrow")


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
    with pytest.raises(ValueError, match="Inconsistent loop orientations detected"):
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

    with pytest.raises(ValueError, match="Expected extension type"):
        spherely.from_geoarrow(arr)


def test_from_geoarrow_invalid_encoding():
    arr = pa.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])

    with pytest.raises(ValueError, match="'geometry_encoding' should be one"):
        spherely.from_geoarrow(arr, geometry_encoding="point")


def test_from_geoarrow_no_arrow_object():
    with pytest.raises(ValueError, match="input should be an Arrow-compatible array"):
        spherely.from_geoarrow(np.array(["POINT (1 1)"], dtype=object))


def test_to_geoarrow():
    arr = spherely.create([1, 2, 3], [1, 2, 3])
    res = spherely.to_geoarrow(
        arr, output_schema=ga.point().with_coord_type(ga.CoordType.INTERLEAVED)
    )
    assert isinstance(res, spherely.ArrowArrayHolder)
    assert hasattr(res, "__arrow_c_array__")

    arr_pa = pa.array(res)
    coords = np.asarray(arr_pa.storage.values)
    expected = np.array([1, 1, 2, 2, 3, 3], dtype="float64")
    np.testing.assert_allclose(coords, expected)


def test_to_geoarrow_wkt():
    arr = spherely.create([1, 2, 3], [1, 2, 3])
    result = pa.array(spherely.to_geoarrow(arr, output_schema=ga.wkt()))
    # TODO assert result
    print(result)


def test_to_geoarrow_wkb():
    arr = spherely.create([1, 2, 3], [1, 2, 3])
    result = pa.array(spherely.to_geoarrow(arr, output_schema=ga.wkb()))
    # TODO assert result
    print(result)


def test_wkt_roundtrip():
    wkt = [
        "POINT (30 10)",
        "LINESTRING (30 10, 10 30, 40 40)",
        "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
        "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
        "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))",
        "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))",
        "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))",
        "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (30 20, 20 15, 20 25, 30 20)))",
        "GEOMETRYCOLLECTION (POINT (40 10), LINESTRING (10 10, 20 20, 10 40), POLYGON ((40 40, 20 45, 45 30, 40 40)))",
    ]

    arr = spherely.from_geoarrow(ga.as_wkt(wkt))
    result = pa.array(spherely.to_geoarrow(arr, output_schema=ga.wkt()))
    np.testing.assert_array_equal(result, wkt)
