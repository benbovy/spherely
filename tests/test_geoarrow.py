import numpy as np

import pytest

import spherely


pa = pytest.importorskip("pyarrow")
ga = pytest.importorskip("geoarrow.pyarrow")


def test_from_geoarrow_wkt() -> None:

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


def test_from_geoarrow_wkb() -> None:

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


def test_from_geoarrow_native() -> None:

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


def test_from_geoarrow_oriented() -> None:
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


def test_from_wkt_planar() -> None:
    arr = ga.as_geoarrow(["LINESTRING (-64 45, 0 45)"])
    result = spherely.from_geoarrow(arr)
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) > 10000.0

    result = spherely.from_geoarrow(arr, planar=True)
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) < 100.0

    result = spherely.from_geoarrow(arr, planar=True, tessellate_tolerance=10)
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) < 10.0


def test_from_geoarrow_projection() -> None:
    arr = ga.as_wkt(["POINT (1 0)", "POINT(0 1)"])

    result = spherely.from_geoarrow(
        arr, projection=spherely.Projection.orthographic(0, 0)
    )
    expected = spherely.points([90, 0], [0, 90])
    # TODO use equality when we support precision / snapping
    # assert spherely.equals(result, expected).all()
    assert (spherely.to_wkt(result) == spherely.to_wkt(expected)).all()


def test_from_geoarrow_no_extension_type() -> None:
    arr = pa.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])

    with pytest.raises(ValueError, match="Expected extension type"):
        spherely.from_geoarrow(arr)


def test_from_geoarrow_invalid_encoding() -> None:
    arr = pa.array(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])

    with pytest.raises(ValueError, match="'geometry_encoding' should be one"):
        spherely.from_geoarrow(arr, geometry_encoding="point")


def test_from_geoarrow_no_arrow_object() -> None:
    with pytest.raises(ValueError, match="input should be an Arrow-compatible array"):
        spherely.from_geoarrow(np.array(["POINT (1 1)"], dtype=object))  # type: ignore


def test_to_geoarrow() -> None:
    arr = spherely.points([1, 2, 3], [1, 2, 3])
    res = spherely.to_geoarrow(
        arr, output_schema=ga.point().with_coord_type(ga.CoordType.INTERLEAVED)
    )
    assert isinstance(res, spherely.ArrowArrayHolder)
    assert hasattr(res, "__arrow_c_array__")

    arr_pa = pa.array(res)
    coords = np.asarray(arr_pa.storage.values)
    expected = np.array([1, 1, 2, 2, 3, 3], dtype="float64")
    np.testing.assert_allclose(coords, expected)


def test_to_geoarrow_wkt() -> None:
    arr = spherely.points([1, 2, 3], [1, 2, 3])
    result = pa.array(spherely.to_geoarrow(arr, output_schema=ga.wkt()))
    expected = pa.array(["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"])
    assert result.storage.equals(expected)


def test_to_geoarrow_wkb() -> None:
    arr = spherely.points([1, 2, 3], [1, 2, 3])
    result = pa.array(spherely.to_geoarrow(arr, output_schema=ga.wkb()))
    # the conversion from lon/lat values to S2 points and back gives some floating
    # point differences, and output to WKB does not do any rounding,
    # therefore checking exact values here
    expected = ga.as_wkb(
        [
            "POINT (0.9999999999999998 1)",
            "POINT (2 1.9999999999999996)",
            "POINT (3.0000000000000004 3.0000000000000004)",
        ]
    )
    assert result.equals(expected)


def test_wkt_roundtrip() -> None:
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


def test_to_geoarrow_no_output_encoding() -> None:
    arr = spherely.points([1, 2, 3], [1, 2, 3])

    with pytest.raises(ValueError, match="'output_schema' should be specified"):
        spherely.to_geoarrow(arr)


def test_to_geoarrow_invalid_output_schema() -> None:
    arr = spherely.points([1, 2, 3], [1, 2, 3])
    with pytest.raises(
        ValueError, match="'output_schema' should be an Arrow-compatible schema"
    ):
        spherely.to_geoarrow(arr, output_schema="WKT")

    with pytest.raises(ValueError, match="Did you pass a valid schema"):
        spherely.to_geoarrow(arr, output_schema=pa.schema([("test", pa.int64())]))


def test_to_geoarrow_projected() -> None:
    arr = spherely.points([1, 2, 3], [1, 2, 3])
    point_schema = ga.point().with_coord_type(ga.CoordType.INTERLEAVED)
    result = pa.array(
        spherely.to_geoarrow(
            arr, output_schema=point_schema, projection=spherely.Projection.lnglat()
        )
    )

    coords = np.asarray(result.storage.values)
    expected = np.array([1, 1, 2, 2, 3, 3], dtype="float64")
    np.testing.assert_allclose(coords, expected)

    # Output to pseudo mercator - generation of expected result
    #   import pyproj
    #   trans = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    #   trans.transform([1, 2, 3], [1, 2, 3])
    result = pa.array(
        spherely.to_geoarrow(
            arr,
            output_schema=point_schema,
            projection=spherely.Projection.pseudo_mercator(),
        )
    )
    coords = np.asarray(result.storage.values)
    expected = np.array(
        [
            111319.49079327357,
            111325.1428663851,
            222638.98158654713,
            222684.20850554405,
            333958.4723798207,
            334111.1714019596,
        ],
        dtype="float64",
    )
    np.testing.assert_allclose(coords, expected)

    # Output to orthographic
    result = pa.array(
        spherely.to_geoarrow(
            arr,
            output_schema=point_schema,
            projection=spherely.Projection.orthographic(0.0, 0.0),
        )
    )
    coords = np.asarray(result.storage.values)
    expected = np.array(
        [0.01744975, 0.01745241, 0.03487824, 0.0348995, 0.05226423, 0.05233596],
        dtype="float64",
    )
    np.testing.assert_allclose(coords, expected, rtol=1e-06)
