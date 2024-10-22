import numpy as np

import spherely


def test_intersects() -> None:
    # test array + scalar
    a = np.array(
        [
            spherely.linestring([(40, 8), (60, 8)]),
            spherely.linestring([(20, 0), (30, 0)]),
        ]
    )
    b = spherely.linestring([(50, 5), (50, 10)])

    actual = spherely.intersects(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.point(50, 8)
    b2 = spherely.point(20, 5)
    assert not spherely.intersects(a2, b2)


def test_equals() -> None:
    # test array + scalar
    a = np.array(
        [
            spherely.linestring([(40, 8), (60, 8)]),
            spherely.linestring([(20, 0), (30, 0)]),
        ]
    )
    b = spherely.point(50, 8)

    actual = spherely.equals(a, b)
    expected = np.array([False, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.point(50, 8)
    b2 = spherely.point(50, 8)
    assert spherely.equals(a2, b2)


def test_contains():
    # test array + scalar
    a = np.array(
        [
            spherely.linestring([(40, 8), (60, 8)]),
            spherely.linestring([(20, 0), (30, 0)]),
        ]
    )
    b = spherely.point(40, 8)

    actual = spherely.contains(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.linestring([(50, 8), (60, 8)])
    b2 = spherely.point(50, 8)
    assert spherely.contains(a2, b2)


def test_contains_polygon():
    # plain vs. hole polygon
    poly_plain = spherely.polygon(shell=[(0, 0), (4, 0), (4, 4), (0, 4)])

    poly_hole = spherely.polygon(
        shell=[(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )

    assert spherely.contains(poly_plain, spherely.point(2, 2))
    assert not spherely.contains(poly_hole, spherely.point(2, 2))


def test_within():
    # test array + scalar
    a = spherely.point(40, 8)
    b = np.array(
        [
            spherely.linestring([(40, 8), (60, 8)]),
            spherely.linestring([(20, 0), (30, 0)]),
        ]
    )

    actual = spherely.within(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.point(50, 8)
    b2 = spherely.linestring([(50, 8), (60, 8)])
    assert spherely.within(a2, b2)


def test_within_polygon():
    # plain vs. hole polygon
    poly_plain = spherely.polygon(shell=[(0, 0), (4, 0), (4, 4), (0, 4)])

    poly_hole = spherely.polygon(
        shell=[(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )

    assert spherely.within(spherely.point(2, 2), poly_plain)
    assert not spherely.within(spherely.point(2, 2), poly_hole)


def test_disjoint():
    a = spherely.point(40, 9)
    b = np.array(
        [
            spherely.linestring([(40, 8), (60, 8)]),
            spherely.linestring([(20, 0), (30, 0)]),
        ]
    )

    actual = spherely.disjoint(a, b)
    expected = np.array([True, True])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.point(50, 9)
    b2 = spherely.linestring([(50, 8), (60, 8)])
    assert spherely.disjoint(a2, b2)


def test_touches():
    a = spherely.Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    b = np.array(
        [
            spherely.Polygon([(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)]),
            spherely.Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]),
        ]
    )

    actual = spherely.touches(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    a_p = spherely.Point(1.0, 1.0)
    b_p = spherely.Point(1.0, 1.0)
    # Points do not have a boundary, so they cannot touch per definition
    # This is consistent with PostGIS for example (cmp. https://postgis.net/docs/ST_Touches.html)
    assert not spherely.touches(a_p, b_p)

    b_line = spherely.LineString([(1.0, 1.0), (1.0, 2.0)])
    assert spherely.touches(a_p, b_line)


def test_covers():
    a = spherely.Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    b_p = np.array(
        [
            spherely.Point(2.0, 2.0),
            spherely.Point(1.0, 1.0),
            spherely.Point(0.5, 0.5),
        ]
    )

    actual = spherely.covers(a, b_p)
    expected = np.array([False, True, True])
    np.testing.assert_array_equal(actual, expected)

    b_poly_in = spherely.Polygon([(0.1, 0.1), (0.1, 0.9), (0.9, 0.9), (0.9, 0.1)])
    b_poly_part_boundary = spherely.Polygon(
        [(0.0, 0.0), (0.0, 0.75), (0.75, 0.75), (0.75, 0.0)]
    )

    assert spherely.covers(a, b_poly_in)
    assert spherely.covers(a, b_poly_part_boundary)  # This fails, but should not


def test_covered_by():
    a = spherely.Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    b_p = np.array(
        [
            spherely.Point(2.0, 2.0),
            spherely.Point(1.0, 1.0),
            spherely.Point(0.5, 0.5),
        ]
    )

    actual = spherely.covered_by(b_p, a)
    expected = np.array([False, True, True])
    np.testing.assert_array_equal(actual, expected)

    b_poly_in = spherely.Polygon([(0.1, 0.1), (0.1, 0.9), (0.9, 0.9), (0.9, 0.1)])
    b_poly_part_boundary = spherely.Polygon(
        [(0.0, 0.0), (0.0, 0.75), (0.75, 0.75), (0.75, 0.0)]
    )

    assert spherely.covered_by(b_poly_in, a)
    assert spherely.covers(b_poly_part_boundary, a)  # This fails, but should not
