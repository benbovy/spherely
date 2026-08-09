import struct
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

import spherely

MakeTree = Callable[[Any], spherely.SpatialIndex]


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


def _make_tree(param: str, geographies: Any) -> spherely.SpatialIndex:
    # build a SpatialIndex directly or round-trip it through
    # encode / from_encoded, with ("fat") or without ("thin") the Geography
    # objects: queries must behave identically on all of them
    tree = spherely.SpatialIndex(geographies)
    if param == "fat":
        tree = spherely.SpatialIndex.from_encoded(tree.encode())
    elif param == "thin":
        tree = spherely.SpatialIndex.from_encoded(
            tree.encode(include_geographies=False)
        )
    return tree


@pytest.fixture(params=["built", "fat", "thin"])
def make_tree(request: pytest.FixtureRequest) -> MakeTree:
    def make(geographies: Any) -> spherely.SpatialIndex:
        return _make_tree(request.param, geographies)

    return make


@pytest.fixture(params=["built", "fat"])
def make_full_tree(request: pytest.FixtureRequest) -> MakeTree:
    # trees that support the full API: predicates, exclusive nearest queries
    # and the geometries property (a thin decoded index raises on those)
    def make(geographies: Any) -> spherely.SpatialIndex:
        return _make_tree(request.param, geographies)

    return make


def test_spatial_index_len(geographies: npt.NDArray[Any], make_tree: MakeTree) -> None:
    tree = make_tree(geographies)
    assert len(tree) == 4


def test_spatial_index_geometries(
    geographies: npt.NDArray[Any], make_full_tree: MakeTree
) -> None:
    tree = make_full_tree(geographies)
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


def test_query_scalar(geographies: npt.NDArray[Any], make_tree: MakeTree) -> None:
    tree = make_tree(geographies)
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    result = tree.query(poly)
    assert result.dtype == np.intp
    # the three nearby points are candidates; the far point is not
    np.testing.assert_array_equal(result, [0, 1, 2])


def test_query_predicate_refines(make_full_tree: MakeTree) -> None:
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
    tree = make_full_tree(geographies)
    triangle = spherely.create_polygon([(0, 0), (5, 0), (0, 5), (0, 0)])

    coarse = tree.query(triangle)
    refined = tree.query(triangle, predicate="contains")

    # the coarse candidate set includes the false positive, the refined
    # (exact) result does not
    np.testing.assert_array_equal(coarse, [0, 1])
    np.testing.assert_array_equal(refined, [0])


def test_query_predicate_intersects(
    geographies: npt.NDArray[Any], make_full_tree: MakeTree
) -> None:
    tree = make_full_tree(geographies)
    point = spherely.create_point(1, 1)
    result = tree.query(point, predicate="intersects")
    np.testing.assert_array_equal(result, [1])


def test_query_array(geographies: npt.NDArray[Any], make_full_tree: MakeTree) -> None:
    tree = make_full_tree(geographies)
    queries = np.array(
        [
            spherely.create_point(1, 1),
            spherely.create_point(50, 50),
        ]
    )
    result = tree.query(queries, predicate="intersects")
    # (input_index, tree_index) pairs
    np.testing.assert_array_equal(result, [[0, 1], [1, 3]])


def test_query_empty_geography_never_returned(make_tree: MakeTree) -> None:
    geoms = np.array(
        [
            spherely.create_point(0, 0),
            spherely.create_polygon(None),  # empty
        ]
    )
    tree = make_tree(geoms)
    assert len(tree) == 2
    poly = spherely.create_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)])
    result = tree.query(poly)
    np.testing.assert_array_equal(result, [0])


def test_query_disjoint_rejected(
    geographies: npt.NDArray[Any], make_full_tree: MakeTree
) -> None:
    tree = make_full_tree(geographies)
    point = spherely.create_point(1, 1)
    with pytest.raises(ValueError, match="disjoint"):
        tree.query(point, predicate="disjoint")  # type: ignore[call-overload]


def test_query_invalid_predicate(
    geographies: npt.NDArray[Any], make_full_tree: MakeTree
) -> None:
    tree = make_full_tree(geographies)
    point = spherely.create_point(1, 1)
    with pytest.raises(ValueError, match="invalid predicate"):
        tree.query(point, predicate="not_a_predicate")  # type: ignore[call-overload]


def test_query_nearest_scalar(
    geographies: npt.NDArray[Any], make_tree: MakeTree
) -> None:
    tree = make_tree(geographies)
    result = tree.query_nearest(spherely.create_point(1.2, 1.2))
    assert result.dtype == np.intp
    np.testing.assert_array_equal(result, [1])


def test_query_nearest_ties(make_tree: MakeTree) -> None:
    # two points exactly symmetric about the query point on the equator
    tree = make_tree(
        [
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
            spherely.create_point(50, 0),
        ]
    )
    query = spherely.create_point(0, 0)
    np.testing.assert_array_equal(tree.query_nearest(query), [0, 1])

    single = tree.query_nearest(query, all_matches=False)
    np.testing.assert_array_equal(single, [0])


def test_query_nearest_ties_lowest_index(make_tree: MakeTree) -> None:
    # more than two geographies tied at the minimum distance: all_matches=False
    # returns the lowest index, not an arbitrary member of the tie
    tree = make_tree(
        [
            spherely.create_point(50, 0),  # far away
            spherely.create_point(2, 0),
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),  # duplicate of index 1
        ]
    )
    query = spherely.create_point(0, 0)

    np.testing.assert_array_equal(tree.query_nearest(query), [1, 2, 3])
    np.testing.assert_array_equal(tree.query_nearest(query, all_matches=False), [1])

    # array form (and with the distances)
    queries = np.array([query, spherely.create_point(49, 0)])
    single, distances = tree.query_nearest(
        queries, all_matches=False, return_distance=True
    )
    np.testing.assert_array_equal(single, [[0, 1], [1, 0]])
    np.testing.assert_allclose(
        distances,
        [
            spherely.distance(query, spherely.create_point(2, 0)),
            spherely.distance(queries[1], spherely.create_point(50, 0)),
        ],
    )


def test_query_nearest_exclusive_ties_lowest_index(make_full_tree: MakeTree) -> None:
    # same, on the exclusive path: the geographies equal to the query are
    # dropped and the lowest of the remaining tie is returned
    tree = make_full_tree(
        [
            spherely.create_point(0, 0),  # equal to the query
            spherely.create_point(2, 0),
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
        ]
    )
    query = spherely.create_point(0, 0)
    np.testing.assert_array_equal(tree.query_nearest(query, exclusive=True), [1, 2, 3])
    np.testing.assert_array_equal(
        tree.query_nearest(query, exclusive=True, all_matches=False), [1]
    )

    # ... and on the "everything at distance zero is equal" fallback path
    tree = make_full_tree(
        [
            spherely.create_point(0, 0),
            spherely.create_point(0, 0),
            spherely.create_point(2, 0),
            spherely.create_point(-2, 0),
        ]
    )
    np.testing.assert_array_equal(tree.query_nearest(query, exclusive=True), [2, 3])
    np.testing.assert_array_equal(
        tree.query_nearest(query, exclusive=True, all_matches=False), [2]
    )


def test_query_nearest_interior_distance_zero(make_tree: MakeTree) -> None:
    # a point inside an indexed polygon is at distance zero from it,
    # consistent with spherely.distance
    poly = spherely.create_polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    tree = make_tree([poly, spherely.create_point(5.1, 5.1)])
    inside = spherely.create_point(5, 5)
    assert spherely.distance(inside, poly) == 0
    np.testing.assert_array_equal(tree.query_nearest(inside), [0])

    # and symmetrically: an indexed point inside a query polygon
    tree = make_tree([spherely.create_point(5, 5), spherely.create_point(20, 20)])
    query_poly = spherely.create_polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    np.testing.assert_array_equal(tree.query_nearest(query_poly), [0])


def test_query_nearest_empty_geographies(make_tree: MakeTree) -> None:
    tree = make_tree(
        [
            spherely.create_polygon(None),  # empty, never returned
            spherely.create_point(10, 10),
        ]
    )
    np.testing.assert_array_equal(tree.query_nearest(spherely.create_point(0, 0)), [1])

    # empty query geography -> no result
    result = tree.query_nearest(spherely.create_polygon(None))
    assert result.shape == (0,)

    # empty tree -> no result
    tree = make_tree(np.array([], dtype=object))
    result = tree.query_nearest(spherely.create_point(0, 0))
    assert result.shape == (0,)


def test_query_nearest_return_distance(
    geographies: npt.NDArray[Any], make_tree: MakeTree
) -> None:
    tree = make_tree(geographies)
    query = spherely.create_point(1.2, 1.2)

    indices, distances = tree.query_nearest(query, return_distance=True)
    np.testing.assert_array_equal(indices, [1])
    assert distances.dtype == np.float64
    np.testing.assert_allclose(distances, spherely.distance(query, geographies[1]))

    # the arguments may also be passed positionally, as in shapely
    positional = tree.query_nearest(query, None, True)
    np.testing.assert_array_equal(positional[0], indices)
    np.testing.assert_array_equal(positional[1], distances)

    # a return_distance only known at runtime is typed as the union of both
    flag = bool(len(indices))
    runtime = tree.query_nearest(query, return_distance=flag)
    assert isinstance(runtime, tuple)

    # radius scales the returned distances
    _, unscaled = tree.query_nearest(query, return_distance=True, radius=1)
    np.testing.assert_allclose(
        unscaled * spherely.EARTH_RADIUS_METERS, distances, rtol=1e-15
    )


def test_query_nearest_array(
    geographies: npt.NDArray[Any], make_tree: MakeTree
) -> None:
    tree = make_tree(geographies)
    queries = np.array(
        [
            spherely.create_point(1.2, 1.2),
            spherely.create_point(49, 49),
        ]
    )

    result = tree.query_nearest(queries)
    assert result.dtype == np.intp
    # (input_index, tree_index) pairs
    np.testing.assert_array_equal(result, [[0, 1], [1, 3]])

    pairs, distances = tree.query_nearest(queries, return_distance=True)
    np.testing.assert_array_equal(pairs, result)
    expected = [
        spherely.distance(queries[0], geographies[1]),
        spherely.distance(queries[1], geographies[3]),
    ]
    np.testing.assert_allclose(distances, expected)


def test_query_nearest_array_ties_and_empty(make_tree: MakeTree) -> None:
    tree = make_tree(
        [
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
            spherely.create_point(50, 0),
        ]
    )
    queries = np.array(
        [
            spherely.create_point(0, 0),  # tied between 0 and 1
            spherely.create_polygon(None),  # empty: contributes no pairs
            spherely.create_point(49, 0),
        ]
    )
    result = tree.query_nearest(queries)
    np.testing.assert_array_equal(result, [[0, 0, 2], [0, 1, 2]])

    single = tree.query_nearest(queries, all_matches=False)
    # ties are broken deterministically on the lowest tree index
    np.testing.assert_array_equal(single, [[0, 2], [0, 2]])


def test_query_nearest_max_distance(
    geographies: npt.NDArray[Any], make_tree: MakeTree
) -> None:
    tree = make_tree(geographies)
    query = spherely.create_point(1.2, 1.2)
    dist = spherely.distance(query, geographies[1])

    np.testing.assert_array_equal(
        tree.query_nearest(query, max_distance=dist * 1.01), [1]
    )
    # inclusive bound
    np.testing.assert_array_equal(tree.query_nearest(query, max_distance=dist), [1])

    # no geography within max_distance -> empty result (shapely semantics)
    result = tree.query_nearest(query, max_distance=1.0)
    assert result.shape == (0,)
    indices, distances = tree.query_nearest(
        query, max_distance=1.0, return_distance=True
    )
    assert indices.shape == (0,)
    assert distances.shape == (0,)

    # array form
    queries = np.array([query, spherely.create_point(49, 49)])
    result = tree.query_nearest(queries, max_distance=dist * 1.01)
    np.testing.assert_array_equal(result, [[0], [1]])

    # array form with no match at all: (2, 0) pairs and 0 distances
    pairs, distances = tree.query_nearest(
        queries, max_distance=1.0, return_distance=True
    )
    assert pairs.shape == (2, 0)
    assert pairs.dtype == np.intp
    assert distances.shape == (0,)
    assert distances.dtype == np.float64


def test_query_nearest_max_distance_inclusive_boundary(make_tree: MakeTree) -> None:
    # the distance returned by query_nearest must itself be accepted as an
    # (inclusive) max_distance: a fixture or two would not catch the ULP-level
    # round trip error between meters and chord angles, so sweep many pairs
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 50)
    lat = rng.uniform(-90, 90, 50)
    tree = make_tree(np.asarray(spherely.points(lon, lat)))

    qlon = rng.uniform(-180, 180, 200)
    qlat = rng.uniform(-90, 90, 200)
    for query in np.asarray(spherely.points(qlon, qlat)):
        expected, distances = tree.query_nearest(query, return_distance=True)
        result = tree.query_nearest(query, max_distance=distances[0])
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("max_distance", [0, -1.0, np.nan, np.inf, -np.inf])
def test_query_nearest_invalid_max_distance(
    geographies: npt.NDArray[Any], make_tree: MakeTree, max_distance: float
) -> None:
    # NaN in particular used to be silently accepted as "no bound"
    tree = make_tree(geographies)
    point = spherely.create_point(1, 1)
    match = "max_distance must be a finite value greater than 0"
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(point, max_distance=max_distance)
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(np.array([point]), max_distance=max_distance)


@pytest.mark.parametrize("radius", [0, -1.0, np.nan, np.inf])
def test_query_nearest_invalid_radius(
    geographies: npt.NDArray[Any], make_tree: MakeTree, radius: float
) -> None:
    # the radius filters here (it scales max_distance): a zero radius would
    # make every distance zero, a negative one would flip the chord angle
    tree = make_tree(geographies)
    point = spherely.create_point(1, 1)
    match = "radius must be a finite value greater than 0"
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(point, radius=radius)
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(point, max_distance=1000.0, radius=radius)
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(np.array([point]), radius=radius)


def test_query_nearest_exclusive(make_full_tree: MakeTree) -> None:
    tree = make_full_tree(
        [
            spherely.create_point(0, 0),
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
        ]
    )
    query = spherely.create_point(0, 0)
    np.testing.assert_array_equal(tree.query_nearest(query), [0])
    result = tree.query_nearest(query, exclusive=True)
    # the equal geography is skipped, the two symmetric neighbors are tied
    np.testing.assert_array_equal(result, [1, 2])

    # a non-equal geography at distance zero is still returned
    query = spherely.create_point(1, 1)
    poly = spherely.create_polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    tree = make_full_tree([spherely.create_point(1, 1), poly])
    result, distances = tree.query_nearest(query, exclusive=True, return_distance=True)
    np.testing.assert_array_equal(result, [1])
    np.testing.assert_array_equal(distances, [0.0])


def test_query_nearest_exclusive_all_equal(make_full_tree: MakeTree) -> None:
    # all geographies at distance zero are equal to the query
    tree = make_full_tree([spherely.create_point(1, 1), spherely.create_point(1, 1)])
    query = spherely.create_point(1, 1)
    result = tree.query_nearest(query, exclusive=True)
    assert result.shape == (0,)

    # with a farther non-equal geography, that one is returned
    far = spherely.create_point(3, 3)
    tree = make_full_tree(
        [spherely.create_point(1, 1), spherely.create_point(1, 1), far]
    )
    indices, distances = tree.query_nearest(query, exclusive=True, return_distance=True)
    np.testing.assert_array_equal(indices, [2])
    np.testing.assert_allclose(distances, spherely.distance(query, far))

    # unless it is beyond max_distance
    result = tree.query_nearest(query, exclusive=True, max_distance=1.0)
    assert result.shape == (0,)


def test_query_nearest_exclusive_all_equal_beyond_batch(
    make_full_tree: MakeTree,
) -> None:
    # more geographies equal to the query than the initial max_results batch
    # of the fallback path (16), so the query has to be re-run with a larger
    # batch until the first result at a distance > 0 shows up
    query = spherely.create_point(1, 1)
    equal = [spherely.create_point(1, 1) for _ in range(40)]
    far = spherely.create_point(3, 3)
    tree = make_full_tree(equal + [far])

    indices, distances = tree.query_nearest(query, exclusive=True, return_distance=True)
    np.testing.assert_array_equal(indices, [len(equal)])
    np.testing.assert_allclose(distances, spherely.distance(query, far))

    # ... and with nothing else in the index, the loop still terminates
    tree = make_full_tree(equal)
    assert tree.query_nearest(query, exclusive=True).shape == (0,)


def test_query_nearest_against_distance_oracle(make_tree: MakeTree) -> None:
    rng = np.random.default_rng(42)
    lon = rng.uniform(-180, 180, 20)
    lat = rng.uniform(-90, 90, 20)
    geographies = np.asarray(spherely.points(lon, lat))
    tree = make_tree(geographies)

    for qlon, qlat in [(0, 0), (179.5, 0.5), (-179.5, -0.5), (0, 89.9), (0, -89.9)]:
        query = spherely.create_point(qlon, qlat)
        result = tree.query_nearest(query)
        distances = np.asarray(spherely.distance(query, geographies))
        expected = np.flatnonzero(distances == distances.min())
        np.testing.assert_array_equal(result, expected)


def test_encode(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    encoded = tree.encode()
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
    # magic, format version and flags (geographies included by default)
    assert encoded[:6] == b"SPIX\x01\x01"
    # encoding is deterministic
    assert tree.encode() == encoded

    thin = tree.encode(include_geographies=False)
    assert thin[:6] == b"SPIX\x01\x00"
    # the thin blob drops the per-geography blocks (and their offset table)
    assert len(thin) < len(encoded)
    assert tree.encode(include_geographies=False) == thin


def test_encode_concurrent() -> None:
    # encode() releases the GIL around the (pure C++) encoding work
    geoms = [spherely.create_point(x, x % 80) for x in range(200)]
    tree = spherely.SpatialIndex(geoms)
    expected = tree.encode()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: tree.encode(), range(8)))
    assert all(result == expected for result in results)


def test_shape_ids_map_to_tree_indices() -> None:
    # the shape id -> tree index map is filled by the constructor rather than
    # by s2geography::GeographyIndex::Add: geographies with several shapes
    # (and ones with none) must still line up with their input positions
    geoms = [
        spherely.create_point(0, 0),
        spherely.create_collection(  # two shapes
            [
                spherely.create_point(30, 30),
                spherely.create_linestring([(31, 31), (32, 32)]),
            ]
        ),
        spherely.create_polygon(None),  # empty: no shape
        spherely.create_point(-60, -60),
    ]
    tree = spherely.SpatialIndex(geoms)
    assert len(tree) == 4

    for i, geog in enumerate(geoms):
        if i == 2:  # the empty geography is indexed but never returned
            continue
        np.testing.assert_array_equal(tree.query(geog, predicate="intersects"), [i])
        np.testing.assert_array_equal(tree.query_nearest(geog), [i])

    # both shapes of the collection resolve to its tree index, and only once
    np.testing.assert_array_equal(
        tree.query(spherely.create_point(31.5, 31.5), predicate="intersects"), []
    )
    np.testing.assert_array_equal(
        tree.query_nearest(spherely.create_point(31.5, 31.5)), [1]
    )


def test_from_encoded_preserves_indices() -> None:
    # multi-shape and empty geographies make the shape id <-> tree index
    # mapping non-trivial: it must survive the encode / from_encoded round
    # trip so that both indexes return the same integer indices
    geoms = [
        spherely.create_point(0, 0),
        spherely.create_collection(  # two shapes
            [
                spherely.create_point(30, 30),
                spherely.create_linestring([(31, 31), (32, 32)]),
            ]
        ),
        spherely.create_polygon(None),  # empty: no shape
        spherely.create_point(-60, -60),
    ]
    tree = spherely.SpatialIndex(geoms)
    decoded = spherely.SpatialIndex.from_encoded(tree.encode())

    assert len(decoded) == len(tree) == 4
    for query in geoms[:2] + geoms[3:] + [spherely.create_point(30.5, 30.5)]:
        np.testing.assert_array_equal(
            decoded.query_nearest(query), tree.query_nearest(query)
        )
        np.testing.assert_array_equal(decoded.query(query), tree.query(query))

    # a decoded index has no shapes of its own to re-encode: encode() hands
    # back the blob it was loaded from, so comparing those bytes with the
    # original would be circular. Check instead that the blob survives another
    # round trip and still answers queries identically.
    redecoded = spherely.SpatialIndex.from_encoded(decoded.encode())
    assert len(redecoded) == len(tree)
    for query in geoms[:2] + geoms[3:] + [spherely.create_point(30.5, 30.5)]:
        np.testing.assert_array_equal(
            redecoded.query_nearest(query), tree.query_nearest(query)
        )
        np.testing.assert_array_equal(redecoded.query(query), tree.query(query))


def test_from_encoded_bytes_like() -> None:
    tree = spherely.SpatialIndex(
        [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    )
    encoded = tree.encode()
    query = spherely.create_point(0.9, 0.9)
    # annotated: an unannotated literal joins to Sequence[int], which
    # from_encoded does not accept
    buffers: list[bytes | bytearray | memoryview] = [
        encoded,
        bytearray(encoded),
        memoryview(encoded),
    ]
    for buffer in buffers:
        decoded = spherely.SpatialIndex.from_encoded(buffer)
        np.testing.assert_array_equal(decoded.query_nearest(query), [1])

    with pytest.raises(TypeError):
        spherely.SpatialIndex.from_encoded(memoryview(encoded)[::2])
    with pytest.raises(TypeError):
        spherely.SpatialIndex.from_encoded("not bytes")  # type: ignore[arg-type]


def test_from_encoded_empty_tree() -> None:
    tree = spherely.SpatialIndex(np.array([], dtype=object))
    decoded = spherely.SpatialIndex.from_encoded(tree.encode())
    assert len(decoded) == 0
    result = decoded.query_nearest(spherely.create_point(0, 0))
    assert result.shape == (0,)
    assert decoded.geometries.shape == (0,)


@pytest.mark.parametrize("include_geographies", [True, False])
def test_from_encoded_is_built(
    geographies: npt.NDArray[Any], include_geographies: bool
) -> None:
    # a decoded index decodes its cells on demand rather than building them,
    # so there is never anything left for build() to force
    tree = spherely.SpatialIndex(geographies)
    decoded = spherely.SpatialIndex.from_encoded(
        tree.encode(include_geographies=include_geographies)
    )
    assert decoded.is_built
    decoded.build()
    decoded.build(tmp_memory_budget=1024)
    assert decoded.is_built
    assert 1 in decoded.query(spherely.create_point(1, 1))

    # the budget is validated before the "nothing to build" early return
    with pytest.raises(ValueError, match="positive"):
        decoded.build(tmp_memory_budget=0)


def test_thin_encoded_unsupported_operations(geographies: npt.NDArray[Any]) -> None:
    # an index loaded from a blob without the geographies cannot support the
    # operations that need the Geography objects; the error points at the
    # include_geographies=True remedy
    tree = spherely.SpatialIndex.from_encoded(
        spherely.SpatialIndex(geographies).encode(include_geographies=False)
    )
    point = spherely.create_point(1, 1)
    match = "not supported for an index loaded.*include_geographies=True"
    with pytest.raises(ValueError, match=match):
        tree.geometries  # noqa: B018
    with pytest.raises(ValueError, match=match):
        tree.query(point, predicate="intersects")
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(point, exclusive=True)
    # including for an (empty) array input, which never reaches the scalar path
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(np.array([], dtype=object), exclusive=True)
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(np.array([point]), exclusive=True)


def test_fat_encoded_full_parity(geographies: npt.NDArray[Any]) -> None:
    # an index loaded from a default (include_geographies=True) blob supports
    # the operations a thin one cannot, with the same results as the built one
    built = spherely.SpatialIndex(geographies)
    tree = spherely.SpatialIndex.from_encoded(built.encode())
    point = spherely.create_point(1, 1)

    np.testing.assert_array_equal(
        tree.query(point, predicate="intersects"),
        built.query(point, predicate="intersects"),
    )
    np.testing.assert_array_equal(
        tree.query_nearest(point, exclusive=True),
        built.query_nearest(point, exclusive=True),
    )
    geoms = tree.geometries
    assert geoms.shape == (4,)
    assert all(spherely.equals(a, b) for a, b in zip(geoms, geographies))


def test_fat_encoded_lazy_reconstruction(geographies: npt.NDArray[Any]) -> None:
    built = spherely.SpatialIndex(geographies)
    assert built._decoded_count == 0

    tree = spherely.SpatialIndex.from_encoded(built.encode())
    # opening and candidate-only queries walk the index cells and
    # reconstruct nothing
    assert tree._decoded_count == 0
    tree.query(spherely.create_point(1, 1))
    assert tree._decoded_count == 0

    # predicate refinement reconstructs (only) the touched candidates, and
    # each entry at most once: the far-away point (50, 50) is never a
    # candidate, so it is never reconstructed
    tree.query(spherely.create_point(1, 1), predicate="intersects")
    touched = tree._decoded_count
    assert 1 <= touched < len(geographies)
    tree.query(spherely.create_point(1, 1), predicate="intersects")
    assert tree._decoded_count == touched

    # geometries materializes every entry
    tree.geometries  # noqa: B018
    assert tree._decoded_count == len(geographies)


def test_from_encoded_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(b"not a spatial index")
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(b"")

    encoded = spherely.SpatialIndex(
        [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    ).encode()
    with pytest.raises(ValueError, match="unsupported encoded SpatialIndex version"):
        spherely.SpatialIndex.from_encoded(encoded[:4] + b"\xff" + encoded[5:])
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(encoded[: len(encoded) // 2])


def _varint(value: int) -> bytes:
    # the (LEB128) varint encoding used by s2's Encoder::put_varint64
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def test_from_encoded_corrupt_value_table() -> None:
    # the header is followed by the geography count, the shape count and one
    # varint value per shape; all three counts are small enough here to be
    # encoded on a single byte each. A thin blob is used so that the spliced
    # bytes below keep targeting the value table (in a blob with geographies
    # the block table would fail validation first).
    geoms = [spherely.create_point(i, 0) for i in range(4)]
    encoded = spherely.SpatialIndex(geoms).encode(include_geographies=False)
    header = 6
    assert encoded[header] == 4  # num_geographies
    assert encoded[header + 1] == 4  # num_shapes (one per point)
    values = header + 2

    # a shape count smaller than the number of encoded shapes leaves the
    # value table too short: queries would read past its end
    truncated = (
        encoded[:header] + _varint(4) + _varint(1) + encoded[values : values + 1]
    )
    truncated += encoded[values + 4 :]
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(truncated)

    # a value outside [0, num_geographies) is not a usable tree index
    out_of_range = bytearray(encoded)
    out_of_range[values] = 100
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(bytes(out_of_range))

    # a huge shape count must be rejected before it is reserved
    huge = encoded[:header] + _varint(4) + _varint(2**40) + encoded[values:]
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(huge)

    # ... as must a huge geography count
    huge = encoded[:header] + _varint(2**63) + encoded[header + 1 :]
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(huge)


def _mixed_geographies(seed: int, num_points: int = 30) -> npt.NDArray[Any]:
    # a mix of every geography kind, including polygons with holes, an
    # antimeridian-crossing polygon, a multi-shape collection and an empty
    # geography
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180, 180, num_points)
    lat = rng.uniform(-85, 85, num_points)
    geoms: list[Any] = list(np.asarray(spherely.points(lon, lat)))
    for x, y in rng.uniform(-80, 80, (5, 2)):
        geoms.append(
            spherely.create_linestring([(x, y), (x + 2, y + 1), (x + 1, y + 3)])
        )
    for x, y in rng.uniform(-60, 60, (5, 2)):
        geoms.append(
            spherely.create_polygon(
                [(x, y), (x + 8, y), (x + 8, y + 8), (x, y + 8)],
                holes=[
                    [(x + 3, y + 3), (x + 5, y + 3), (x + 5, y + 5), (x + 3, y + 5)]
                ],
            )
        )
    geoms.append(
        # crosses the antimeridian
        spherely.create_polygon([(178, -5), (-178, -5), (-178, 5), (178, 5)])
    )
    geoms.append(
        spherely.create_collection(
            [
                spherely.create_point(40, 40),
                spherely.create_linestring([(41, 41), (42, 42)]),
            ]
        )
    )
    geoms.append(spherely.create_polygon(None))  # empty
    return np.array(geoms)


@pytest.mark.parametrize("seed", [7, 21])
def test_fat_encoded_randomized_parity(seed: int) -> None:
    # a fat-decoded index must answer every query through the same code paths
    # as the built index it came from: results must match exactly
    geoms = _mixed_geographies(seed)
    built = spherely.SpatialIndex(geoms)
    fat = spherely.SpatialIndex.from_encoded(built.encode())

    queries = [
        spherely.create_point(0, 0),
        spherely.create_polygon([(-10, -10), (30, -10), (30, 30), (-10, 30)]),
        # probes the antimeridian-crossing polygon from both sides
        spherely.create_point(179.5, 0),
        spherely.create_point(-179.5, 0),
        spherely.create_polygon([(175, -8), (-175, -8), (-175, 8), (175, 8)]),
        geoms[0],
    ]
    predicates = [
        "intersects",
        "within",
        "contains",
        "covers",
        "covered_by",
        "touches",
        "equals",
    ]
    for query in queries:
        np.testing.assert_array_equal(built.query(query), fat.query(query))
        for predicate in predicates:
            np.testing.assert_array_equal(
                built.query(query, predicate=predicate),  # type: ignore[call-overload]
                fat.query(query, predicate=predicate),  # type: ignore[call-overload]
            )
        np.testing.assert_array_equal(
            built.query_nearest(query), fat.query_nearest(query)
        )
        for exclusive in [False, True]:
            b_idx, b_dist = built.query_nearest(
                query, exclusive=exclusive, return_distance=True
            )
            f_idx, f_dist = fat.query_nearest(
                query, exclusive=exclusive, return_distance=True
            )
            np.testing.assert_array_equal(b_idx, f_idx)
            np.testing.assert_array_equal(b_dist, f_dist)


def test_fat_encoded_geometries_round_trip() -> None:
    # geometries reconstructs every entry in insertion order, preserving the
    # geography type (e.g. a one-point MULTIPOINT stays a MULTIPOINT) and
    # empty geographies
    geoms = np.array(
        [
            spherely.create_point(10, 10),
            spherely.create_multipoint([(20, 20)]),  # one-point multipoint
            spherely.create_polygon(None),  # empty
            spherely.create_collection([spherely.create_point(30, 30)]),
            spherely.create_polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        ]
    )
    tree = spherely.SpatialIndex.from_encoded(spherely.SpatialIndex(geoms).encode())
    result = tree.geometries
    assert result.shape == geoms.shape
    np.testing.assert_array_equal(
        spherely.get_type_id(result), spherely.get_type_id(geoms)
    )
    np.testing.assert_array_equal(spherely.is_empty(result), spherely.is_empty(geoms))
    for a, b in zip(result, geoms):
        if not spherely.is_empty(b):
            assert spherely.equals(a, b)
    # the array is cached: a second access returns the same objects
    assert result[0] is tree.geometries[0]


def test_fat_encoded_lazy_large_tree() -> None:
    # opening a large fat blob and running candidate-only queries must not
    # reconstruct any geography at all
    rng = np.random.default_rng(3)
    geoms = np.asarray(
        spherely.points(rng.uniform(-180, 180, 500), rng.uniform(-85, 85, 500))
    )
    tree = spherely.SpatialIndex.from_encoded(spherely.SpatialIndex(geoms).encode())
    assert tree._decoded_count == 0
    tree.query(spherely.create_polygon([(0, 0), (20, 0), (20, 20), (0, 20)]))
    assert tree._decoded_count == 0

    # a predicate query only reconstructs its candidates
    candidates = tree.query(
        spherely.create_polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    )
    tree.query(
        spherely.create_polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        predicate="contains",
    )
    assert tree._decoded_count == len(candidates)

    # a distance query reconstructs the entries whose shapes it examines --
    # a local neighborhood of the target, not the whole tree. The bound is a
    # near-constant one on purpose: a regression to a reconstruction count
    # that grows with the tree has to fail here (the count is 43 as written,
    # against 500 entries)
    tree.query_nearest(spherely.create_point(12, 34))
    assert tree._decoded_count <= 64


def _fat_layout(num_geographies: int, num_shapes: int) -> tuple[int, int]:
    # (table_start, blocks_start) of a fat blob whose counts and values all
    # fit single-byte varints
    table = 6 + 2 + num_shapes
    return table, table + 16 * num_geographies


def test_from_encoded_unknown_flags() -> None:
    encoded = spherely.SpatialIndex([spherely.create_point(0, 0)]).encode()
    corrupt = encoded[:5] + bytes([encoded[5] | 0x02]) + encoded[6:]
    with pytest.raises(ValueError, match="unsupported encoded SpatialIndex flags"):
        spherely.SpatialIndex.from_encoded(corrupt)


def test_from_encoded_corrupt_block_table() -> None:
    geoms = [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    encoded = spherely.SpatialIndex(geoms).encode()
    table, blocks = _fat_layout(2, 2)
    off0, len0, off1, len1 = struct.unpack_from("<QQQQ", encoded, table)
    assert (off0, off1) == (0, len0)

    def patched(entry: int, offset: int, length: int) -> bytes:
        out = bytearray(encoded)
        struct.pack_into("<QQ", out, table + 16 * entry, offset, length)
        return bytes(out)

    # truncated offset table
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(encoded[: table + 8])
    # block reaching past the end of the blob
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(patched(1, off1, 2**32))
    # overlapping blocks (offset rewound onto the previous block)
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(patched(1, 0, len1))
    # mis-sized block leaving a gap before the next one
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(patched(0, 0, len0 - 1))
    # runt block, too small to hold the type/empty bytes and a tag
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(patched(0, 0, 3))


def test_from_encoded_non_monotonic_value_table() -> None:
    # a fat blob's shape -> entry map must be one contiguous ascending run
    # per entry (shape addressing within a geography relies on it)
    geoms = [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    encoded = spherely.SpatialIndex(geoms).encode()
    # header (6 bytes) + the two single-byte counts, then one value per shape
    values = 8
    assert encoded[values : values + 2] == b"\x00\x01"
    corrupt = bytearray(encoded)
    corrupt[values : values + 2] = b"\x01\x00"
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(bytes(corrupt))


def test_from_encoded_corrupt_geography_block() -> None:
    geoms = [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    encoded = spherely.SpatialIndex(geoms).encode()
    _, blocks = _fat_layout(2, 2)

    # a block that no longer decodes as a geography fails cleanly on first
    # touch (the open itself stays lazy and does not notice)
    corrupt = bytearray(encoded)
    corrupt[blocks + 2] = 0xFF  # clobber the s2geography tag of block 0
    tree = spherely.SpatialIndex.from_encoded(bytes(corrupt))
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex geography"):
        tree.geometries  # noqa: B018

    # ... as does an out-of-range geography type byte
    corrupt = bytearray(encoded)
    corrupt[blocks] = 100
    tree = spherely.SpatialIndex.from_encoded(bytes(corrupt))
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex geography"):
        tree.geometries  # noqa: B018


def test_from_encoded_unwrappable_geography_kind() -> None:
    # the tagged encoding's kind byte selects the s2geography subclass to
    # build: the kinds a spherely Geography cannot wrap (here SHAPE_INDEX)
    # must be rejected rather than cached in a slot and handed out by
    # ``geometries``, where every accessor raises "Unsupported Geography
    # subclass" (and an ENCODED_SHAPE_INDEX would outlive its own buffer)
    geoms = [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    encoded = spherely.SpatialIndex(geoms).encode()
    _, blocks = _fat_layout(2, 2)

    corrupt = bytearray(encoded)
    corrupt[blocks + 2] = 5  # s2geography::GeographyKind::SHAPE_INDEX
    tree = spherely.SpatialIndex.from_encoded(bytes(corrupt))
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex geography"):
        tree.geometries  # noqa: B018


def test_from_encoded_block_type_byte_mismatch() -> None:
    # nothing implies the block's type byte from its tagged encoding: without
    # a cross-check a relabelled block yields a geography whose get_type_id
    # and WKT disagree
    geoms = [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    encoded = spherely.SpatialIndex(geoms).encode()
    _, blocks = _fat_layout(2, 2)
    assert encoded[blocks] == 0  # GeographyType::Point

    # LineString, Polygon, MultiLineString, MultiPolygon, GeometryCollection
    # and None: none of them can describe a POINT-kind block
    for geog_type in [1, 2, 4, 5, 6, 0xFF]:
        corrupt = bytearray(encoded)
        corrupt[blocks] = geog_type
        tree = spherely.SpatialIndex.from_encoded(bytes(corrupt))
        with pytest.raises(ValueError, match="invalid encoded SpatialIndex geography"):
            tree.geometries  # noqa: B018

    # a singular type and its MULTI* form share a kind, so swapping the two
    # labels is not detectable from the block alone -- the geometry itself is
    # unaffected (see type_matches_kind)
    corrupt = bytearray(encoded)
    corrupt[blocks] = 3  # GeographyType::MultiPoint
    tree = spherely.SpatialIndex.from_encoded(bytes(corrupt))
    assert spherely.to_wkt(tree.geometries[0]) == "POINT (0 0)"


def test_from_encoded_corrupt_block_touched_by_query_nearest(
    geographies: npt.NDArray[Any],
) -> None:
    # a corrupt block first touched from inside query_nearest -- i.e. from the
    # ShapeFactory, deep inside s2's distance search rather than from spherely
    # code -- surfaces as the same clean ValueError, and leaves the index
    # usable: the entries decoded before it stay cached, candidate queries
    # (which reconstruct nothing) keep working, and a repeated distance query
    # raises identically rather than doing something new
    built = spherely.SpatialIndex(geographies)
    encoded = built.encode()
    table, blocks = _fat_layout(4, 4)
    offset, _ = struct.unpack_from("<QQ", encoded, table + 16 * 2)

    corrupt = bytearray(encoded)
    corrupt[blocks + offset + 2] = 0xFF  # clobber entry 2's s2geography tag
    tree = spherely.SpatialIndex.from_encoded(bytes(corrupt))

    point = spherely.create_point(1, 1)
    np.testing.assert_array_equal(tree.query(point), built.query(point))
    assert tree._decoded_count == 0

    with pytest.raises(ValueError, match="invalid encoded SpatialIndex geography"):
        tree.query_nearest(point)
    # entries 0 and 1 were reconstructed on the way to the bad one
    assert tree._decoded_count == 2

    np.testing.assert_array_equal(tree.query(point), built.query(point))
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex geography"):
        tree.query_nearest(point)
    assert tree._decoded_count == 2


def test_fat_encoded_concurrent_queries() -> None:
    # a fully decoded fat index is read-only from here on: concurrent queries
    # must agree with the single-threaded answers and reconstruct nothing more
    geoms = _mixed_geographies(11)
    built = spherely.SpatialIndex(geoms)
    tree = spherely.SpatialIndex.from_encoded(built.encode())
    tree.geometries  # noqa: B018
    decoded = tree._decoded_count
    assert decoded == len(geoms)

    queries = [
        spherely.create_point(0, 0),
        spherely.create_polygon([(-10, -10), (30, -10), (30, 30), (-10, 30)]),
        spherely.create_point(179.5, 0),
        spherely.create_linestring([(20, 20), (25, 25)]),
    ]
    expected = [
        (
            built.query(q, predicate="intersects"),
            built.query_nearest(q),
            built.query_nearest(q, exclusive=True),
        )
        for q in queries
    ]

    def run(i: int) -> list[tuple[Any, Any, Any]]:
        return [
            (
                tree.query(q, predicate="intersects"),
                tree.query_nearest(q),
                tree.query_nearest(q, exclusive=True),
            )
            for q in queries
        ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))

    for got in results:
        for (a, b, c), (x, y, z) in zip(got, expected):
            np.testing.assert_array_equal(a, x)
            np.testing.assert_array_equal(b, y)
            np.testing.assert_array_equal(c, z)
    assert tree._decoded_count == decoded


def _collection_geographies() -> npt.NDArray[Any]:
    # collections mixing all three dimensions: their entries own several
    # shapes each, which is what exercises the shape id -> (entry, sub-shape)
    # addressing of a fat blob
    geoms: list[Any] = []
    for x, y in [(0, 0), (20, 20), (-40, 10), (60, -30)]:
        geoms.append(
            spherely.create_collection(
                [
                    spherely.create_point(x, y),
                    spherely.create_linestring([(x + 1, y), (x + 2, y + 1)]),
                    spherely.create_polygon(
                        [(x + 3, y), (x + 5, y), (x + 5, y + 2), (x + 3, y + 2)]
                    ),
                ]
            )
        )
    geoms.append(spherely.create_collection([spherely.create_point(80, 80)]))
    geoms.append(spherely.create_collection([]))  # empty collection
    geoms.append(spherely.create_point(0.5, 0.5))  # inside the first collection
    return np.array(geoms)


def test_collections_full_parity() -> None:
    geoms = _collection_geographies()
    built = spherely.SpatialIndex(geoms)
    fat = spherely.SpatialIndex.from_encoded(built.encode())
    thin = spherely.SpatialIndex.from_encoded(built.encode(include_geographies=False))

    queries = [
        spherely.create_point(0, 0),
        spherely.create_point(4, 1),
        spherely.create_linestring([(1, 0), (2, 1)]),
        spherely.create_polygon([(-1, -1), (6, -1), (6, 3), (-1, 3)]),
        geoms[0],
        geoms[-1],
    ]
    predicates = [
        "intersects",
        "within",
        "contains",
        "covers",
        "covered_by",
        "touches",
        "equals",
    ]
    for query in queries:
        # candidate sets and distance queries agree on all three flavors
        for other in [fat, thin]:
            np.testing.assert_array_equal(built.query(query), other.query(query))
            b_idx, b_dist = built.query_nearest(query, return_distance=True)
            o_idx, o_dist = other.query_nearest(query, return_distance=True)
            np.testing.assert_array_equal(b_idx, o_idx)
            np.testing.assert_array_equal(b_dist, o_dist)
        # predicates and exclusive nearest need the geographies (thin raises)
        for predicate in predicates:
            np.testing.assert_array_equal(
                built.query(query, predicate=predicate),  # type: ignore[call-overload]
                fat.query(query, predicate=predicate),  # type: ignore[call-overload]
            )
        b_idx, b_dist = built.query_nearest(query, exclusive=True, return_distance=True)
        f_idx, f_dist = fat.query_nearest(query, exclusive=True, return_distance=True)
        np.testing.assert_array_equal(b_idx, f_idx)
        np.testing.assert_array_equal(b_dist, f_dist)


def test_from_encoded_block_shape_count_mismatch() -> None:
    # entry 0 has one shape (a point), entry 1 has none (an empty polygon);
    # swapping their blocks (and table lengths, keeping the section
    # well-formed) must be caught by the shape count cross-check on touch
    geoms = [spherely.create_point(0, 0), spherely.create_polygon(None)]
    encoded = spherely.SpatialIndex(geoms).encode()
    table, blocks = _fat_layout(2, 1)
    off0, len0, off1, len1 = struct.unpack_from("<QQQQ", encoded, table)

    out = bytearray(encoded)
    struct.pack_into("<QQQQ", out, table, 0, len1, len1, len0)
    block0 = encoded[blocks + off0 : blocks + off0 + len0]
    block1 = encoded[blocks + off1 : blocks + off1 + len1]
    out[blocks : blocks + len0 + len1] = block1 + block0

    tree = spherely.SpatialIndex.from_encoded(bytes(out))
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex geography"):
        tree.geometries  # noqa: B018


def test_encode_flavor_mismatch_on_decoded_index() -> None:
    geoms = [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    fat_bytes = spherely.SpatialIndex(geoms).encode()
    thin_bytes = spherely.SpatialIndex(geoms).encode(include_geographies=False)

    fat = spherely.SpatialIndex.from_encoded(fat_bytes)
    thin = spherely.SpatialIndex.from_encoded(thin_bytes)
    # a decoded index re-encodes to the exact blob it was loaded from
    assert fat.encode() == fat_bytes
    assert thin.encode(include_geographies=False) == thin_bytes
    # ... and cannot fabricate the other flavor
    with pytest.raises(ValueError, match="include_geographies"):
        fat.encode(include_geographies=False)
    with pytest.raises(ValueError, match="include_geographies"):
        thin.encode()


def _short_id_map(num_geographies: int) -> bytes:
    # a fat blob of ``num_geographies`` single-shape points whose header claims
    # one shape fewer than the index structure it still carries verbatim: the
    # id map no longer covers the last shape id the index cells reference
    geoms = [spherely.create_point(i, 0) for i in range(num_geographies)]
    encoded = spherely.SpatialIndex(geoms).encode()
    assert encoded[6] == num_geographies  # num_geographies
    assert encoded[7] == num_geographies  # num_shapes (one per point)
    values = 8
    assert encoded[values : values + num_geographies] == bytes(range(num_geographies))
    return (
        encoded[:6]
        + bytes([num_geographies, num_geographies - 1])
        + encoded[values : values + num_geographies - 1]
        + encoded[values + num_geographies :]
    )


def test_from_encoded_id_map_shorter_than_index() -> None:
    # the shape count check in from_encoded compares the id map with
    # num_shape_ids(), which for a fat blob is the id map's own size: without
    # a cross-check against the index cells this blob is accepted, and the
    # shape id its cells reference past the end of the map is then read out of
    # bounds inside the distance queries (a segfault, not an exception)
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(_short_id_map(4))


def test_from_encoded_id_map_longer_than_index() -> None:
    # the mirror image: an id map with more (still monotonic) entries than the
    # index has shapes makes num_shape_ids() over-report, so it is rejected too
    geoms = [spherely.create_point(i, 0) for i in range(3)]
    encoded = spherely.SpatialIndex(geoms).encode()
    values = 8
    assert encoded[values : values + 3] == b"\x00\x01\x02"
    corrupt = (
        encoded[:6]
        + bytes([3, 4])
        + b"\x00\x01\x02\x02"  # one extra entry, still ascending
        + encoded[values + 3 :]
    )
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(corrupt)


def test_from_encoded_shape_id_check_is_not_a_decode() -> None:
    # the cell walk that checks the shape ids must not reconstruct anything:
    # opening a fat blob stays lazy
    geoms = np.asarray(spherely.points(np.arange(-80.0, 80.0), np.zeros(160)))
    tree = spherely.SpatialIndex.from_encoded(spherely.SpatialIndex(geoms).encode())
    assert tree._decoded_count == 0


def test_from_encoded_huge_geography_count_not_allocated() -> None:
    # a 12 byte fat blob claiming 2**31-1 geographies must be rejected on the
    # counts alone: sizing the per-geography vectors on it first would ask for
    # tens of gigabytes before any bounds check ran
    corrupt = b"SPIX" + bytes([1, 1]) + _varint(2**31 - 1) + _varint(0)
    assert len(corrupt) < 16
    with pytest.raises(ValueError, match="invalid encoded SpatialIndex"):
        spherely.SpatialIndex.from_encoded(corrupt)
