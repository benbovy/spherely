import numpy as np
import pytest
from packaging.version import Version

import spherely


def test_from_wkt():
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


polygon_with_bad_hole_wkt = (
    "POLYGON "
    "((20 35, 10 30, 10 10, 30 5, 45 20, 20 35),"
    "(30 20, 20 25, 20 15, 30 20))"
)


@pytest.mark.skipif(
    Version(spherely.__s2geography_version__) < Version("0.2.0"),
    reason="Needs s2geography >= 0.2.0",
)
def test_from_wkt_oriented():
    # by default re-orients the inner ring
    result = spherely.from_wkt(polygon_with_bad_hole_wkt)
    assert (
        str(result)
        == "POLYGON ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (20 15, 20 25, 30 20, 20 15))"
    )

    # if we force to not orient, we get an error
    with pytest.raises(RuntimeError, match="Inconsistent loop orientations detected"):
        spherely.from_wkt(polygon_with_bad_hole_wkt, oriented=True)


@pytest.mark.skipif(
    Version(spherely.__s2geography_version__) < Version("0.2.0"),
    reason="Needs s2geography >= 0.2.0",
)
def test_from_wkt_planar():
    result = spherely.from_wkt("LINESTRING (-64 45, 0 45)")
    assert spherely.distance(result, spherely.point(-30.1, 45)) > 10000

    result = spherely.from_wkt("LINESTRING (-64 45, 0 45)", planar=True)
    assert spherely.distance(result, spherely.point(-30.1, 45)) < 100

    result = spherely.from_wkt(
        "LINESTRING (-64 45, 0 45)", planar=True, tessellate_tolerance=10
    )
    assert spherely.distance(result, spherely.point(-30.1, 45)) < 10


@pytest.mark.skipif(
    Version(spherely.__s2geography_version__) >= Version("0.2.0"),
    reason="Needs s2geography < 0.2.0",
)
def test_from_wkt_unsupported_keywords():

    with pytest.raises(ValueError):
        spherely.from_wkt(polygon_with_bad_hole_wkt, oriented=True)

    with pytest.raises(ValueError):
        spherely.from_wkt("LINESTRING (-64 45, 0 45)", planar=True)


def test_to_wkt():
    arr = spherely.points([1.1, 2, 3], [1.1, 2, 3])
    result = spherely.to_wkt(arr)
    expected = np.array(["POINT (1.1 1.1)", "POINT (2 2)", "POINT (3 3)"], dtype=object)
    np.testing.assert_array_equal(result, expected)


def test_to_wkt_precision():
    arr = spherely.points([0.12345], [0.56789])
    result = spherely.to_wkt(arr)
    assert result[0] == "POINT (0.12345 0.56789)"

    result = spherely.to_wkt(arr, precision=2)
    assert result[0] == "POINT (0.12 0.57)"
