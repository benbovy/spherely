# Plan: `SpatialIndex` for spherely (issue #72)

## Context

spherely (S2-backed analog of shapely) currently has no way to spatially index a
large collection of `Geography` objects. Every pairwise predicate (`intersects`,
`contains`, …) is computed brute-force via `py::vectorize`, which is O(N·M) for a
spatial join. Issue #72 asks for an STRtree-like index so candidate pairs can be
pre-filtered cheaply, as shapely's `STRtree` does.

**Feasibility: confirmed — all primitives already exist in the pinned
`s2geography 0.2.0` (conda-forge), so no upstream changes are needed.** Key facts
from exploration:

- `s2geography::GeographyIndex` (header-only, `s2geography/index.h`) wraps a
  `MutableS2ShapeIndex` and is purpose-built as "a GEOSSTRTree index" over a
  *vector* of geographies. Public API (verified at the `0.2.0` tag):
  - `void Add(const Geography& geog, int value)` — `value` is the caller's array index.
  - `int value(int shape_id) const` — maps an internal shape id back to that array index.
  - `const MutableS2ShapeIndex& ShapeIndex() const`.
  - nested `Iterator` with `void Query(const std::vector<S2CellId>& covering, std::unordered_set<int>* out)`
    which fills `out` with matching array indices (cell-overlap candidates).
- `s2geography::Geography::Region() -> std::unique_ptr<S2Region>` and
  `GetCellUnionBound(std::vector<S2CellId>*)` exist, so we can compute a query
  covering via `S2RegionCoverer`.
- The R `s2` package implements exactly this pattern (`s2-matrix.cpp`): build
  `GeographyIndex`, then per query feature `coverer.GetCovering(*Geog().Region(), &cells)`
  → `iterator.Query(cells, &indices)`.
- spherely's existing predicate functions (`s2geog::s2_intersects`, `s2_contains`,
  etc., used in `src/predicates.cpp`) operate on `ShapeIndexGeography` and can be
  reused directly for the refinement step, via the lazy
  `Geography::geog_index()` accessor (`src/geography.hpp:82`).

**Decided scope (per user):** ship the index + `query()` with optional predicate
refinement. Distance/nearest (`S2ClosestEdgeQuery`) and serialization are
explicitly deferred to a follow-up PR. Class name: **`SpatialIndex`**.

## Target Python API

```python
tree = spherely.SpatialIndex(geographies)   # iterable / object-ndarray of Geography
len(tree)                                    # number of indexed geographies
tree.geometries                              # object-ndarray of the inputs (same order)

# scalar query -> sorted np.intp array of tree indices
idx = tree.query(geom)                       # coarse: cells overlap (candidate set)
idx = tree.query(geom, predicate="intersects")  # refined

# array query -> shape (2, K): row 0 = input index, row 1 = tree index (shapely-compatible)
pairs = tree.query(np.array([...]), predicate="intersects")
```

- `predicate=None` → coarse cell-overlap candidates (superset of true intersections).
- Supported `predicate` values (all are subsets of the intersecting-candidate set,
  so refinement is correct): `"intersects"`, `"within"`, `"contains"`, `"covers"`,
  `"covered_by"`, `"touches"`, `"equals"`. Semantics match
  shapely: a tree geom `t` matches when `predicate(query_geom, t)` is True.
- `"disjoint"` is **rejected** in v1 (it is *not* a subset of the overlap candidate
  set; computing it via the index needs the complement — defer). Raise `ValueError`.

## Implementation

### 1. New C++ source: `src/spatial_index.cpp`
Mirror the structure of `src/predicates.cpp` (a `void init_spatial_index(py::module&)`
entry point, `namespace s2geog = s2geography`, reuse `PyObjectGeography`).

Includes: `<s2geography.h>`, `<s2geography/index.h>`, `<s2/s2region_coverer.h>`,
`<s2/s2cell_union.h>`, `"geography.hpp"`, `"pybind11.hpp"`.

Define `class SpatialIndex`:
- **Members:** `std::unique_ptr<s2geog::GeographyIndex> m_index;` and
  `py::array m_geographies;` (an object-dtype ndarray held to (a) keep the indexed
  `Geography` C++ objects alive — `GeographyIndex` only borrows their `S2Shape`s —
  and (b) back the `geometries` property).
- **Constructor `SpatialIndex(py::array_t<PyObjectGeography> geographies)`:**
  store the array; `m_index = std::make_unique<s2geog::GeographyIndex>();` then loop
  `i` over the flat array, `auto* g = geographies.data()[i]->as_geog_ptr();`
  `m_index->Add(g->geog(), static_cast<int>(i));`. Empty geographies contribute no
  shapes (documented: they are never returned). Validate 1-D input; reject `None`
  entries with a clear `TypeError` (reuse `PyObjectGeography::check_type`).
- **`__len__`** → number of input geographies.
- **`geometries` property** → return `m_geographies`.
- **Private helper `query_one(Geography* query_geog, predicate) -> std::vector<int>`:**
  1. `auto region = query_geog->geog().Region();`
  2. `S2RegionCoverer coverer;` (default options for v1) →
     `std::vector<S2CellId> covering; coverer.GetCovering(*region, &covering);`
  3. `std::unordered_set<int> hits; GeographyIndex::Iterator it(m_index.get); it.Query(covering, &hits);`
  4. If `predicate != None`: keep only `c` where
     `predfn(query_geog->geog_index(), candidate->geog_index(), options)` is true,
     where `candidate = m_geographies.data()[c]->as_geog_ptr()`. Map predicate name →
     callable over two `ShapeIndexGeography` (see step 2 reuse note below).
  5. copy set → `std::vector<int>`, `std::sort`.
- **`query(scalar Geography)`** → `py::array_t<std::intp_t>` (sorted).
- **`query(array of Geography)`** → build two `std::vector<int>` (input idx, tree idx)
  and return a `(2, K)` `py::array_t<std::intp_t>` (shapely's `query` bulk layout).

### 2. Reuse predicate functions (small shared helper)
The predicate lambdas currently live *inside* `init_predicates` in
`src/predicates.cpp` and aren't reachable. Add a tiny shared mapping so both files
use one source of truth. Lightest option:

- Add `src/predicates.hpp` declaring a factory, e.g.
  `using PredicateFunc = std::function<bool(const s2geog::ShapeIndexGeography&, const s2geog::ShapeIndexGeography&)>;`
  and `PredicateFunc get_predicate(const std::string& name);` returning a closure
  over the right `s2geog::s2_intersects`/`s2_contains` call + the appropriate
  `S2BooleanOperation::Options` (e.g. closed model for `covers`/`covered_by`,
  the dual-options trick for `touches`, argument-swap for `within`/`covered_by`).
- Implement it in `predicates.cpp` (or a new `predicates_common.cpp`) and have
  `init_predicates` register the public vectorized predicates by reusing the same
  closures, avoiding duplication. `SpatialIndex::query_one` calls `get_predicate`.
- Unknown / unsupported name (incl. `"disjoint"`) → throw `py::value_error`.

### 3. Wire-up
- `src/spherely.cpp`: declare `void init_spatial_index(py::module&);` and call it in
  `PYBIND11_MODULE` (next to `init_predicates`).
- `CMakeLists.txt`: add `src/spatial_index.cpp` (and `predicates_common.cpp` if
  split out) to `CPP_SOURCES`. No new link deps — `GeographyIndex` is header-only and
  `s2geography`/`s2::s2` are already linked. Gate on s2geography ≥ 0.2.0 if needed
  (already the project floor: `s2geography >=0.2.0,<0.3`).

### 4. Type stubs — `src/spherely.pyi`
Add (near the predicates section):
```python
class SpatialIndex:
    def __init__(self, geographies: Iterable[Geography]) -> None: ...
    def __len__(self) -> int: ...
    @property
    def geometries(self) -> T_NDArray_Geography: ...
    @overload
    def query(self, geography: Geography, predicate: str | None = None) -> npt.NDArray[np.intp]: ...
    @overload
    def query(self, geography: Iterable[Geography], predicate: str | None = None) -> npt.NDArray[np.intp]: ...
```

### 5. Docs
- `docs/api.rst`: new "Spatial indexing" section listing `SpatialIndex`.
- `docs/api_hidden.rst`: add the class' members so autosummary picks them up
  (match the existing hidden-entry pattern).

### 6. Tests — `tests/test_spatial_index.py`
Mirror `tests/test_predicates.py` conventions (typed, numpy assertions). Cover:
- build + `len()` + `geometries` round-trips and dtype/order.
- coarse `query(scalar)` returns candidate indices (sorted, np.intp).
- `query(scalar, predicate="intersects")` refines vs coarse (strict subset case).
- each supported predicate gives shapely-equivalent results on a small hand-checked
  fixture (points/lines/polygons spanning a couple of cells).
- array query returns `(2, K)` `(input_idx, tree_idx)` layout.
- empty geography in the tree is never returned; empty query returns empty.
- `predicate="disjoint"` and unknown predicate raise; `None` entry raises `TypeError`.

## Verification

```bash
pixi run -e test compile            # build the extension (configure+compile tasks)
pixi run -e test tests              # run pytest, incl. new test_spatial_index.py
pixi run -e lint mypy               # type-check stubs against tests
```
Manual smoke test in `pixi shell -e test`:
```python
import numpy as np, spherely
geoms = spherely.points([0,1,2,50], [0,1,2,50])
t = spherely.SpatialIndex(geoms)
assert len(t) == 4
q = spherely.create_polygon([(-1,-1),(3,-1),(3,3),(-1,3)])
print(t.query(q))                       # -> [0 1 2]
print(t.query(q, predicate="contains")) # refined subset
```
Cross-check a handful of results against shapely's `STRtree` on the planar
equivalents to confirm the join semantics line up.

## Deferred to follow-up (out of scope here)
- `query_nearest` / `nearest` / distance filtering via `S2ClosestEdgeQuery`
  (+ `S2ClosestEdgeQuery::ShapeIndexTarget`, header `<s2/s2closest_edge_query.h>`).
- Index serialization / pickling (S2's `EncodedS2ShapeIndex` enables mmap'd,
  shippable indexes — noted as high value in issue #72).
- Configurable coverer (`max_cells`) and a dedicated point-only fast path
  (`S2PointIndex`) for large point sets.
```
