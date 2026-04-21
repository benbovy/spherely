"""Concurrency tests for the GIL-released boolean operations.

Paired with the change that wraps ``s2geog::s2_boolean_operation`` in
``py::gil_scoped_release`` inside ``BooleanOp::operator()``. These tests
exercise the failure modes most likely to show up under a race:

1. concurrent calls on **shared input Geographies** (lazy index / shadow
   cache reads),
2. concurrent calls on **freshly constructed Geographies** (first-access
   lazy materialization),
3. **mixed operations** (union/intersection/difference/sym_diff) over
   shared inputs,
4. Python-side object churn happening **in parallel** with the s2 work
   (stress on the release/re-acquire interleave),
5. a performance canary that fails closed if the GIL is *not* being
   released (serial ≈ threaded means the release didn't land).

Each correctness test produces a **serial reference** first, then fans
the same inputs out across a ThreadPoolExecutor and compares bit-for-bit
via ``numpy.testing.assert_array_equal`` — no tolerance, because if the
GIL release is broken we expect either crashes, data corruption, or
cache-pollution-induced output drift, not rounding.

Run under ThreadSanitizer in CI to catch data races that don't surface
as wrong answers on any single run:

    CFLAGS="-fsanitize=thread" pip install -e . --no-build-isolation
    pytest tests/test_boolean_operations_concurrency.py
"""

from __future__ import annotations

import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import spherely


# Keep the workload big enough that the GIL-release window is long enough
# for a race to actually interleave, but small enough to keep test runtime
# under a few seconds per case.
N_SRC_LAT = 18
N_SRC_LON = 36
N_TGT_LAT = 9
N_TGT_LON = 18
N_THREADS = min(os.cpu_count() or 1, 4)


# --- fixtures ---------------------------------------------------------------


def _cell_polygons(ny: int, nx: int, lat_span=(-85.0, 85.0), lon_span=(-175.0, 175.0)):
    """Return an (n_cells,) object array of spherely.PolygonGeography."""
    lat_edges = np.linspace(lat_span[0], lat_span[1], ny + 1)
    lon_edges = np.linspace(lon_span[0], lon_span[1], nx + 1)
    polys = np.empty(ny * nx, dtype=object)
    for j in range(ny):
        y0, y1 = float(lat_edges[j]), float(lat_edges[j + 1])
        for i in range(nx):
            x0, x1 = float(lon_edges[i]), float(lon_edges[i + 1])
            shell = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            polys[j * nx + i] = spherely.create_polygon(shell, oriented=True)
    return polys


def _pair_arrays():
    """Build ``(dst_array, src_array)`` pairs covering the regridding workload.

    Every target cell is paired with every source cell (full outer product —
    not representative of the real STRtree-filtered pair list, but guarantees
    deterministic, well-defined pair arrays of known size).
    """
    src = _cell_polygons(N_SRC_LAT, N_SRC_LON)
    tgt = _cell_polygons(N_TGT_LAT, N_TGT_LON, lat_span=(-80.0, 80.0))
    dst_idx, src_idx = np.meshgrid(np.arange(len(tgt)), np.arange(len(src)), indexing="ij")
    return tgt[dst_idx.ravel()], src[src_idx.ravel()]


@pytest.fixture(scope="module")
def shared_pairs():
    return _pair_arrays()


# --- correctness under concurrency ------------------------------------------


def _call_op(op, dst, src):
    return spherely.area(op(dst, src), radius=1.0)


def _fan_out(op, dst, src, n_threads=N_THREADS, iters_per_thread=1):
    """Run ``op(dst, src)`` from ``n_threads`` threads simultaneously. Returns
    a list of numpy arrays, one per thread invocation."""
    barrier = threading.Barrier(n_threads)

    def work(_):
        barrier.wait()  # maximise simultaneous entry
        results = []
        for _ in range(iters_per_thread):
            results.append(_call_op(op, dst, src))
        return results

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        all_results = list(pool.map(work, range(n_threads)))
    return [r for thread_results in all_results for r in thread_results]


@pytest.mark.parametrize(
    "op",
    [spherely.intersection, spherely.union, spherely.difference, spherely.symmetric_difference],
)
def test_concurrent_shared_inputs_match_serial(op, shared_pairs):
    """N threads calling the same op on the same input arrays should return
    bit-identical area arrays, matching a serial reference."""
    dst, src = shared_pairs
    serial = _call_op(op, dst, src)

    results = _fan_out(op, dst, src)
    for i, r in enumerate(results):
        np.testing.assert_array_equal(
            r,
            serial,
            err_msg=f"threaded result #{i} diverged from serial for {op.__name__}",
        )


def test_concurrent_lazy_index_race():
    """Fresh Geographies — the lazy index has not been materialized yet.

    All worker threads hit exactly the same (a, b) pair as their first access,
    maximising the chance of a race inside the first call to ``geog_index()``
    or any shadow cache built on first boolean-op invocation.
    """
    for _ in range(5):  # repeat so one bad interleave is likely to surface
        a = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
        b = spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))")
        dst = np.array([a] * 200, dtype=object)
        src = np.array([b] * 200, dtype=object)
        serial = spherely.area(spherely.intersection(dst, src), radius=1.0)

        # Now drop the references that `serial` forced to materialize and
        # redo with fresh geographies for the concurrent path.
        a2 = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
        b2 = spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))")
        dst2 = np.array([a2] * 200, dtype=object)
        src2 = np.array([b2] * 200, dtype=object)
        threaded = _fan_out(spherely.intersection, dst2, src2, n_threads=N_THREADS)
        for t in threaded:
            np.testing.assert_array_equal(t, serial)


def test_concurrent_mixed_operations(shared_pairs):
    """Different ops running concurrently on the same input pair should not
    interfere — each op's output matches its own serial reference."""
    dst, src = shared_pairs
    ops = [
        spherely.intersection,
        spherely.union,
        spherely.difference,
        spherely.symmetric_difference,
    ]
    expected = {op: _call_op(op, dst, src) for op in ops}

    barrier = threading.Barrier(len(ops))

    def work(op):
        barrier.wait()
        return op, _call_op(op, dst, src)

    with ThreadPoolExecutor(max_workers=len(ops)) as pool:
        for op, got in pool.map(work, ops):
            np.testing.assert_array_equal(
                got,
                expected[op],
                err_msg=f"mixed-ops concurrency corrupted output of {op.__name__}",
            )


# --- stress interleaves -----------------------------------------------------


def test_intersection_with_parallel_python_churn():
    """While the s2 op runs with the GIL released, another thread is actively
    churning Python objects (allocating / dropping references, triggering GC).
    This stresses the release / re-acquire boundary: if the release drops
    the GIL but ``make_py_geography`` on re-acquire races with GC running on
    another thread, we'd see crashes or refcount corruption.

    Uses a smaller pair array than the shared_pairs fixture so churner and
    intersection threads don't saturate every core for minutes."""
    # ~2500 pairs — small enough that the full fan_out finishes in ~1 s,
    # large enough that the GIL-release window actually overlaps churn.
    src = _cell_polygons(9, 18)
    tgt = _cell_polygons(5, 10, lat_span=(-80.0, 80.0))
    dst_idx, src_idx = np.meshgrid(
        np.arange(len(tgt)), np.arange(len(src)), indexing="ij"
    )
    dst = tgt[dst_idx.ravel()]
    src_arr = src[src_idx.ravel()]
    serial = _call_op(spherely.intersection, dst, src_arr)

    stop = threading.Event()

    def churner():
        # Non-spinning churn: allocate, sleep briefly to yield the GIL,
        # gc.collect every few iterations. Without the sleep the churner
        # monopolises the GIL and starves the intersection threads, which
        # turns the test into a livelock probe rather than a correctness one.
        i = 0
        while not stop.is_set():
            bag = [[j] * 8 for j in range(200)]
            _ = {j: bag[j][0] for j in range(0, 200, 7)}
            if i % 5 == 0:
                gc.collect()
            i += 1
            time.sleep(0.001)

    churn_threads = [threading.Thread(target=churner) for _ in range(2)]
    for t in churn_threads:
        t.start()
    try:
        results = _fan_out(spherely.intersection, dst, src_arr, n_threads=N_THREADS)
    finally:
        stop.set()
        for t in churn_threads:
            t.join()
    for r in results:
        np.testing.assert_array_equal(r, serial)


def test_refcounts_stable_after_concurrent_runs():
    """After many concurrent boolean-op calls on shared inputs, the refcount
    on those inputs should be back to its pre-call baseline. Leaked references
    here would indicate an incref was done under the GIL but the matching
    decref was skipped (or vice versa) during the release window."""
    a = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
    b = spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))")
    import sys
    dst = np.array([a] * 50, dtype=object)
    src = np.array([b] * 50, dtype=object)
    rc_a_before = sys.getrefcount(a)
    rc_b_before = sys.getrefcount(b)
    for _ in range(20):
        _fan_out(spherely.intersection, dst, src, n_threads=N_THREADS)
    # Allow GC to settle any transient references from the ThreadPool.
    gc.collect()
    assert sys.getrefcount(a) == rc_a_before, "ref leak on input Geography a"
    assert sys.getrefcount(b) == rc_b_before, "ref leak on input Geography b"


# --- performance canary -----------------------------------------------------


@pytest.mark.slow
def test_gil_release_actually_enables_parallelism():
    """This is the one test that *directly* confirms the GIL was released.

    If spherely has not been rebuilt with the gil_scoped_release patch, the
    threaded path will serialize on the GIL and run no faster than a single
    thread. We use an intentionally expensive workload — many pairs that
    *actually* overlap so each intersection does real s2 work — so the
    threading win is well above machine-noise.
    """
    if N_THREADS < 2:
        pytest.skip("need at least 2 cores for a meaningful speedup measurement")

    # A pair of large partially-overlapping polygons, repeated. Each
    # intersection exercises the full s2 boolean-op path (not a bbox
    # rejection), so per-call time is in the 100µs-1ms range.
    big1 = spherely.from_wkt(
        "POLYGON ((-80 -40, 80 -40, 80 40, -80 40, -80 -40))"
    )
    big2 = spherely.from_wkt(
        "POLYGON ((-40 -80, 40 -80, 40 80, -40 80, -40 -80))"
    )
    n = 4000
    dst = np.array([big1] * n, dtype=object)
    src = np.array([big2] * n, dtype=object)

    def serial():
        spherely.intersection(dst, src)

    def threaded():
        splits = np.array_split(np.arange(n), N_THREADS)

        def worker(idx):
            spherely.intersection(dst[idx], src[idx])

        with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
            list(pool.map(worker, splits))

    def median_of(fn, trials=3):
        serial()  # warm any caches
        samples = []
        for _ in range(trials):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return sorted(samples)[len(samples) // 2]

    t_serial = median_of(serial)
    t_threaded = median_of(threaded)
    # Expect at least 1.3× on 4 cores. Well below the theoretical 4× to
    # tolerate bench noise, but any number <1.1 means the release almost
    # certainly did not land.
    assert t_threaded < t_serial / 1.3, (
        f"threaded={t_threaded * 1000:.1f}ms was not meaningfully faster than "
        f"serial={t_serial * 1000:.1f}ms — GIL release may not be in effect"
    )
