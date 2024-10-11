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


def test_predicates_polygon():
    # plain vs. hole polygon
    poly_plain = spherely.polygon(shell=[(0, 0), (4, 0), (4, 4), (0, 4)])

    poly_hole = spherely.polygon(
        shell=[(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )

    assert spherely.contains(poly_plain, spherely.point(2, 2))
    assert not spherely.contains(poly_hole, spherely.point(2, 2))
