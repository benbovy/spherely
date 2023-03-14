import numpy as np

import spherely


def test_intersects() -> None:
    # test array + scalar
    a = np.array(
        [
            spherely.LineString([(40, 8), (60, 8)]),
            spherely.LineString([(20, 0), (30, 0)]),
        ]
    )
    b = spherely.LineString([(50, 5), (50, 10)])

    actual = spherely.intersects(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.Point(50, 8)
    b2 = spherely.Point(20, 5)
    assert not spherely.intersects(a2, b2)


def test_equals() -> None:
    # test array + scalar
    a = np.array(
        [
            spherely.LineString([(40, 8), (60, 8)]),
            spherely.LineString([(20, 0), (30, 0)]),
        ]
    )
    b = spherely.Point(50, 8)

    actual = spherely.equals(a, b)
    expected = np.array([False, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.Point(50, 8)
    b2 = spherely.Point(50, 8)
    assert spherely.equals(a2, b2)


def test_contains():
    # test array + scalar
    a = np.array(
        [
            spherely.LineString([(40, 8), (60, 8)]),
            spherely.LineString([(20, 0), (30, 0)]),
        ]
    )
    b = spherely.Point(40, 8)

    actual = spherely.contains(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.LineString([(50, 8), (60, 8)])
    b2 = spherely.Point(50, 8)
    assert spherely.contains(a2, b2)


def test_within():
    # test array + scalar
    a = spherely.Point(40, 8)
    b = np.array(
        [
            spherely.LineString([(40, 8), (60, 8)]),
            spherely.LineString([(20, 0), (30, 0)]),
        ]
    )

    actual = spherely.within(a, b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.Point(50, 8)
    b2 = spherely.LineString([(50, 8), (60, 8)])
    assert spherely.within(a2, b2)


def test_disjoint():
    a = spherely.Point(40, 9)
    b = np.array(
        [
            spherely.LineString([(40, 8), (60, 8)]),
            spherely.LineString([(20, 0), (30, 0)]),
        ]
    )

    actual = spherely.disjoint(a, b)
    expected = np.array([True, True])
    np.testing.assert_array_equal(actual, expected)

    # two scalars
    a2 = spherely.Point(50, 9)
    b2 = spherely.LineString([(50, 8), (60, 8)])
    assert spherely.disjoint(a2, b2)
