from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import spherely


def _pairs(result: npt.NDArray[np.intp]) -> set[tuple[Any, ...]]:
    # the (2, N) result of an array query as a set of (input, tree) pairs
    return set(map(tuple, result.T))


@pytest.fixture
def geographies() -> npt.NDArray[Any]:
    # three points near (0, 0)..(2, 2) and one far-away point
    return np.array(
        [
            spherely.create_point(0, 0),
            spherely.create_point(1, 1),
            spherely.create_point(2, 2),
            spherely.create_point(50, 50),
        ]
    )


def test_spatial_index_len(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert len(tree) == 4


def test_spatial_index_geometries(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    geoms = tree.geometries
    assert isinstance(geoms, np.ndarray)
    assert geoms.shape == (4,)
    assert all(spherely.equals(a, b) for a, b in zip(geoms, geographies))


def test_spatial_index_from_list() -> None:
    tree = spherely.SpatialIndex(
        [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    )
    assert len(tree) == 2


def test_spatial_index_copies_input_array() -> None:
    geoms = np.array([spherely.create_point(0, 0), spherely.create_point(1, 1)])
    tree = spherely.SpatialIndex(geoms)
    # replacing an element of the input array does not affect the index
    geoms[0] = spherely.create_point(40, 40)
    result = tree.query(spherely.create_point(0, 0), predicate="intersects")
    np.testing.assert_array_equal(result, [0])
    assert spherely.equals(tree.geometries[0], spherely.create_point(0, 0))


def test_is_built(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert tree.is_built is False
    tree.build()
    assert tree.is_built is True


def test_is_built_after_query(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert tree.is_built is False
    # the first query builds the index
    tree.query(spherely.create_point(1, 1))
    assert tree.is_built is True


def test_build_is_idempotent(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert not tree.is_built
    tree.build()
    assert tree.is_built
    tree.build()
    # a budget override on an already-built index is accepted and ignored
    tree.build(tmp_memory_budget=1)
    assert tree.is_built

    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])
    np.testing.assert_array_equal(tree.query(poly), [0, 1, 2])


def test_build_after_query(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    expected = tree.query(point, predicate="intersects")
    # the query already built the index; building again changes nothing
    assert tree.is_built
    tree.build()
    assert tree.is_built
    np.testing.assert_array_equal(tree.query(point, predicate="intersects"), expected)


def test_build_does_not_change_results(geographies: npt.NDArray[Any]) -> None:
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    lazy = spherely.SpatialIndex(geographies)
    eager = spherely.SpatialIndex(geographies)
    eager.build()
    assert eager.is_built
    assert not lazy.is_built

    np.testing.assert_array_equal(eager.query(poly), lazy.query(poly))
    np.testing.assert_array_equal(
        eager.query(poly, predicate="contains"), lazy.query(poly, predicate="contains")
    )


# the s2 default and two raised budgets -- raising is the only useful
# direction, see the SpatialIndex.build docstring
@pytest.mark.parametrize("budget", [104_857_600, 1024**3, 4 * 1024**3])
def test_build_budget_does_not_change_matches(
    geographies: npt.NDArray[Any], budget: int
) -> None:
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    default = spherely.SpatialIndex(geographies)
    default.build()
    tuned = spherely.SpatialIndex(geographies)
    tuned.build(tmp_memory_budget=budget)
    assert tuned.is_built

    # the budget moves s2's batch boundaries and with them the subdivision of
    # the index, so the unrefined candidate set is only guaranteed to stay a
    # superset of the matches -- what is invariant is the refined answer
    matches = tuned.query(poly, predicate="contains")
    np.testing.assert_array_equal(matches, default.query(poly, predicate="contains"))
    assert set(matches) <= set(tuned.query(poly))


def test_build_budget_below_default_is_legal(geographies: npt.NDArray[Any]) -> None:
    # a budget far below the default is accepted and still correct; it only
    # makes the build slower (more batches), so it is not a value to copy
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    default = spherely.SpatialIndex(geographies)
    default.build()
    tiny = spherely.SpatialIndex(geographies)
    tiny.build(tmp_memory_budget=1)
    assert tiny.is_built

    matches = tiny.query(poly, predicate="contains")
    np.testing.assert_array_equal(matches, default.query(poly, predicate="contains"))
    assert set(matches) <= set(tiny.query(poly))


def _overlapping_geographies() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # 600 overlapping 30-gons and 50 polygon queries over the same patch: big
    # enough that the coarse candidate set holds real false positives, and that
    # a budget below the default splits the build into several batches (18000
    # edges at s2geometry's 226 temporary bytes per edge is about 4 MB)
    rng = np.random.default_rng(7)

    def polygon(
        cx: float, cy: float, radius: float, nvertices: int
    ) -> spherely.Geography:
        angles = np.linspace(0, 2 * np.pi, nvertices, endpoint=False)
        coords = [
            (cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles.tolist()
        ]
        return spherely.create_polygon(coords + [coords[0]])

    indexed = np.array(
        [polygon(rng.uniform(-1, 1), rng.uniform(-1, 1), 0.3, 30) for _ in range(600)]
    )
    queries = np.array(
        [polygon(rng.uniform(-1, 1), rng.uniform(-1, 1), 0.2, 8) for _ in range(50)]
    )
    return indexed, queries


def test_build_budget_does_not_change_matches_on_larger_collection() -> None:
    indexed, queries = _overlapping_geographies()

    default = spherely.SpatialIndex(indexed)
    default.build()
    expected = default.query(queries, predicate="intersects")

    # unlike the four-geography fixture above, here the coarse candidate set is
    # strictly larger than the matches, so "superset of the matches" is a real
    # constraint rather than an accident of the collection's size
    assert _pairs(expected) < _pairs(default.query(queries))

    # 1 MiB forces the build into several batches (this collection needs about
    # 4 MB of temporary space) while 4 GiB builds it in one, as the default
    # does here: the candidate sets need not agree, the matches do
    for budget in (1024**2, 4 * 1024**3):
        tuned = spherely.SpatialIndex(indexed)
        tuned.build(tmp_memory_budget=budget)
        matches = tuned.query(queries, predicate="intersects")
        np.testing.assert_array_equal(matches, expected)
        assert _pairs(matches) <= _pairs(tuned.query(queries))


@pytest.mark.parametrize("budget", [0, -1, -(1024**3)])
def test_build_invalid_budget(geographies: npt.NDArray[Any], budget: int) -> None:
    tree = spherely.SpatialIndex(geographies)
    with pytest.raises(ValueError, match="positive"):
        tree.build(tmp_memory_budget=budget)
    assert not tree.is_built

    # the failed call left the index buildable
    tree.build()
    assert tree.is_built
    np.testing.assert_array_equal(
        tree.query(spherely.create_point(1, 1), predicate="intersects"), [1]
    )

    # the budget is validated before the "already built" early return, so an
    # invalid value is rejected even when there is nothing left to build
    with pytest.raises(ValueError, match="positive"):
        tree.build(tmp_memory_budget=budget)


def test_build_budget_out_of_range(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    # the budget is a C int64; anything larger is rejected by the binding
    with pytest.raises(TypeError):
        tree.build(tmp_memory_budget=2**63)
    assert not tree.is_built


def test_build_budget_is_restored(geographies: npt.NDArray[Any]) -> None:
    # the override is scoped to a single build: the process-wide s2 setting is
    # back to its previous value once the call returns
    before = spherely._s2_tmp_memory_budget()

    tuned = spherely.SpatialIndex(geographies)
    # deliberately different from the budget in effect, so that a failure to
    # put the previous value back is visible here
    tuned.build(tmp_memory_budget=before + 4096)
    assert spherely._s2_tmp_memory_budget() == before

    # ...so an index built afterwards without one behaves exactly as if the
    # override had never happened
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])
    plain = spherely.SpatialIndex(geographies)
    plain.build()
    assert spherely._s2_tmp_memory_budget() == before
    np.testing.assert_array_equal(
        plain.query(poly, predicate="contains"),
        tuned.query(poly, predicate="contains"),
    )


@pytest.fixture
def restore_s2_budget() -> Iterator[None]:
    # for the tests below, which move the process-wide s2 setting themselves
    before = spherely._s2_tmp_memory_budget()
    yield
    spherely._set_s2_tmp_memory_budget(before)


def test_s2_tmp_memory_budget_round_trip(restore_s2_budget: None) -> None:
    # the accessors reach the real s2 flag: what is set is what is read back
    spherely._set_s2_tmp_memory_budget(4096)
    assert spherely._s2_tmp_memory_budget() == 4096
    spherely._set_s2_tmp_memory_budget(8192)
    assert spherely._s2_tmp_memory_budget() == 8192


def test_build_budget_restores_previous_value(
    geographies: npt.NDArray[Any], restore_s2_budget: None
) -> None:
    # test_build_budget_is_restored runs at s2geometry's default, where putting
    # the previous value back and resetting to the default are the same thing;
    # start from a non-default value so that only the former passes
    spherely._set_s2_tmp_memory_budget(4096)

    tree = spherely.SpatialIndex(geographies)
    tree.build(tmp_memory_budget=8192)
    assert tree.is_built
    assert spherely._s2_tmp_memory_budget() == 4096


def test_build_empty_index() -> None:
    tree = spherely.SpatialIndex([])
    # nothing was queued, so there is nothing to build
    assert tree.is_built
    tree.build(tmp_memory_budget=1024)
    assert len(tree) == 0
    np.testing.assert_array_equal(tree.query(spherely.create_point(0, 0)), [])


def test_query_scalar(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    result = tree.query(poly)
    assert result.dtype == np.intp
    # the three nearby points are candidates; the far point is not
    np.testing.assert_array_equal(result, [0, 1, 2])


def test_query_predicate_refines() -> None:
    # point 1 is outside the triangle (beyond the hypotenuse) but close enough
    # that it falls within the triangle's coarse cell covering; point 2 is far
    # away and is not a candidate at all
    geographies = np.array(
        [
            spherely.create_point(1, 1),
            spherely.create_point(4, 4),
            spherely.create_point(50, 50),
        ]
    )
    tree = spherely.SpatialIndex(geographies)
    triangle = spherely.create_polygon([(0, 0), (5, 0), (0, 5), (0, 0)])

    coarse = tree.query(triangle)
    refined = tree.query(triangle, predicate="contains")

    # the coarse candidate set includes the false positive, the refined
    # (exact) result does not
    np.testing.assert_array_equal(coarse, [0, 1])
    np.testing.assert_array_equal(refined, [0])


def test_query_predicate_intersects(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    result = tree.query(point, predicate="intersects")
    np.testing.assert_array_equal(result, [1])


def test_query_array(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    queries = np.array(
        [
            spherely.create_point(1, 1),
            spherely.create_point(50, 50),
        ]
    )
    result = tree.query(queries, predicate="intersects")
    # (input_index, tree_index) pairs
    np.testing.assert_array_equal(result, [[0, 1], [1, 3]])


def test_query_empty_geography_never_returned() -> None:
    geoms = np.array(
        [
            spherely.create_point(0, 0),
            spherely.create_polygon(None),  # empty
        ]
    )
    tree = spherely.SpatialIndex(geoms)
    assert len(tree) == 2
    poly = spherely.create_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)])
    result = tree.query(poly)
    np.testing.assert_array_equal(result, [0])


def test_query_disjoint_rejected(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    with pytest.raises(ValueError, match="disjoint"):
        tree.query(point, predicate="disjoint")  # type: ignore[call-overload]


def test_query_invalid_predicate(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    with pytest.raises(ValueError, match="invalid predicate"):
        tree.query(point, predicate="not_a_predicate")  # type: ignore[call-overload]
