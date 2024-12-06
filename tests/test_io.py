import struct

import numpy as np
import pytest

import spherely


def test_from_wkt() -> None:
    result = spherely.from_wkt(["POINT (1 1)", "POINT(2 2)", "POINT(3 3)"])
    expected = spherely.points([1, 2, 3], [1, 2, 3])
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


def test_from_wkt_invalid() -> None:
    # TODO can we provide better error type?
    with pytest.raises(RuntimeError):
        spherely.from_wkt(["POINT (1)"])


def test_from_wkt_wrong_type() -> None:
    with pytest.raises(TypeError, match="expected bytes, int found"):
        spherely.from_wkt([1])  # type: ignore

    # TODO support missing values
    with pytest.raises(TypeError, match="expected bytes, NoneType found"):
        spherely.from_wkt(["POINT (1 1)", None])  # type: ignore


polygon_with_bad_hole_wkt = (
    "POLYGON "
    "((20 35, 10 30, 10 10, 30 5, 45 20, 20 35),"
    "(30 20, 20 25, 20 15, 30 20))"
)


def test_from_wkt_oriented() -> None:
    # by default re-orients the inner ring
    result = spherely.from_wkt(polygon_with_bad_hole_wkt)
    assert (
        str(result)
        == "POLYGON ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (20 15, 20 25, 30 20, 20 15))"
    )

    # if we force to not orient, we get an error
    with pytest.raises(RuntimeError, match="Inconsistent loop orientations detected"):
        spherely.from_wkt(polygon_with_bad_hole_wkt, oriented=True)


def test_from_wkt_planar() -> None:
    result = spherely.from_wkt("LINESTRING (-64 45, 0 45)")
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) > 10000

    result = spherely.from_wkt("LINESTRING (-64 45, 0 45)", planar=True)
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) < 100

    result = spherely.from_wkt(
        "LINESTRING (-64 45, 0 45)", planar=True, tessellate_tolerance=10
    )
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) < 10


def test_to_wkt() -> None:
    arr = spherely.points([1.1, 2, 3], [1.1, 2, 3])
    result = spherely.to_wkt(arr)
    expected = np.array(["POINT (1.1 1.1)", "POINT (2 2)", "POINT (3 3)"], dtype=object)
    np.testing.assert_array_equal(result, expected)


def test_to_wkt_precision() -> None:
    arr = spherely.points([0.12345], [0.56789])
    result = spherely.to_wkt(arr)
    assert result[0] == "POINT (0.12345 0.56789)"

    result = spherely.to_wkt(arr, precision=2)
    assert result[0] == "POINT (0.12 0.57)"


POINT11_WKB = struct.pack("<BI2d", 1, 1, 1.0, 1.0)
NAN = struct.pack("<d", float("nan"))
POINT_NAN_WKB = struct.pack("<BI", 1, 1) + (NAN * 2)
MULTIPOINT_NAN_WKB = struct.pack("<BII", 1, 4, 1) + POINT_NAN_WKB
GEOMETRYCOLLECTION_NAN_WKB = struct.pack("<BII", 1, 7, 1) + POINT_NAN_WKB
INVALID_WKB = bytes.fromhex(
    "01030000000100000002000000507daec600b1354100de02498e5e3d41306ea321fcb03541a011a53d905e3d41"
)  # noqa: E501


def test_from_wkb_point_empty() -> None:
    result = spherely.from_wkb([POINT11_WKB, POINT_NAN_WKB, MULTIPOINT_NAN_WKB])
    # empty MultiPoint is converted to empty Point
    expected = spherely.from_wkt(["POINT (1 1)", "POINT EMPTY", "POINT EMPTY"])
    assert spherely.equals(result, expected).all()

    result2 = spherely.from_wkb(GEOMETRYCOLLECTION_NAN_WKB)
    assert str(result2) == "GEOMETRYCOLLECTION (POINT EMPTY)"


def test_from_wkb_invalid() -> None:
    with pytest.raises(RuntimeError, match="Expected endian byte"):
        spherely.from_wkb(b"")

    with pytest.raises(RuntimeError):
        spherely.from_wkb([b"\x01\x01\x00\x00\x00\x00"])

    # TODO should this raise an error?
    # with pytest.raises(RuntimeError):
    result = spherely.from_wkb(INVALID_WKB)
    assert str(result) == "POLYGON ((108.7761 -10.2852, 108.7761 -10.2852))"


def test_from_wkb_invalid_type() -> None:
    with pytest.raises(TypeError, match="expected bytes, str found"):
        spherely.from_wkb("POINT (1 1)")  # type: ignore


@pytest.mark.parametrize(
    "geog",
    [
        spherely.create_point(45, 50),
        spherely.create_multipoint([(5, 50), (6, 51)]),
        spherely.create_linestring([(5, 50), (6, 51)]),
        spherely.create_multilinestring([[(5, 50), (6, 51)], [(15, 60), (16, 61)]]),
        spherely.create_polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
        # with hole
        spherely.create_polygon(
            shell=[(5, 60), (6, 60), (6, 50), (5, 50)],
            holes=[[(5.1, 59), (5.9, 59), (5.9, 51), (5.1, 51)]],
        ),
        spherely.create_multipolygon(
            [
                spherely.create_polygon([(5, 50), (5, 60), (6, 60), (6, 51)]),
                spherely.create_polygon([(10, 100), (10, 160), (11, 160), (11, 100)]),
            ]
        ),
        spherely.create_collection([spherely.create_point(40, 50)]),
        spherely.create_collection(
            [
                spherely.create_point(0, 0),
                spherely.create_linestring([(0, 0), (1, 1)]),
                spherely.create_polygon([(0, 0), (1, 0), (1, 1)]),
            ]
        ),
    ],
)
def test_wkb_roundtrip(geog) -> None:
    wkb = spherely.to_wkb(geog)
    result = spherely.from_wkb(wkb)
    # roundtrip through Geography unit vector is not exact, so equals can fail
    # TODO properly test this once `equals` supports snapping/precision
    # assert spherely.equals(result, geog)
    assert str(result) == str(geog)


def test_from_wkb_oriented() -> None:
    # WKB for POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0)) -> non-CCW box
    wkb = bytes.fromhex(
        "010300000001000000050000000000000000000000000000000000000000000000000000000000000000002440000000000000244000000000000024400000000000002440000000000000000000000000000000000000000000000000"
    )  # noqa: E501

    result = spherely.from_wkb(wkb)
    # by default re-oriented to take the smaller polygon
    assert str(result) == "POLYGON ((10 0, 10 10, 0 10, 0 0, 10 0))"
    assert spherely.within(spherely.create_point(5, 5), result)

    result = spherely.from_wkb(wkb, oriented=True)
    assert str(result) == "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"
    assert not spherely.within(spherely.create_point(5, 5), result)


def test_from_wkb_planar() -> None:
    wkb = spherely.to_wkb(spherely.from_wkt("LINESTRING (-64 45, 0 45)"))

    result = spherely.from_wkb(wkb)
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) > 10000

    result = spherely.from_wkb(wkb, planar=True)
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) < 100

    result = spherely.from_wkb(wkb, planar=True, tessellate_tolerance=10)
    assert spherely.distance(result, spherely.create_point(-30.1, 45)) < 10
