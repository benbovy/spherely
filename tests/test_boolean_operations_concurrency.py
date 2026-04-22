import gc
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import spherely


N_SRC_LAT = 18
N_SRC_LON = 36
N_TGT_LAT = 9
N_TGT_LON = 18
N_THREADS = min(os.cpu_count() or 1, 4)


def _cell_polygons(ny, nx, lat_span=(-85.0, 85.0), lon_span=(-175.0, 175.0)):
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


@pytest.fixture(scope="module")
def shared_pairs():
    src = _cell_polygons(N_SRC_LAT, N_SRC_LON)
    tgt = _cell_polygons(N_TGT_LAT, N_TGT_LON, lat_span=(-80.0, 80.0))
    dst_idx, src_idx = np.meshgrid(
        np.arange(len(tgt)), np.arange(len(src)), indexing="ij"
    )
    return tgt[dst_idx.ravel()], src[src_idx.ravel()]


def _call_op(op, dst, src):
    return spherely.area(op(dst, src), radius=1.0)


def _fan_out(op, dst, src, n_threads=N_THREADS, iters_per_thread=1):
    """Run ``op(dst, src)`` from ``n_threads`` threads simultaneously."""
    barrier = threading.Barrier(n_threads)

    def work(_):
        barrier.wait()
        results = []
        for _ in range(iters_per_thread):
            results.append(_call_op(op, dst, src))
        return results

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        all_results = list(pool.map(work, range(n_threads)))
    return [r for thread_results in all_results for r in thread_results]


@pytest.mark.parametrize(
    "op",
    [
        spherely.intersection,
        spherely.union,
        spherely.difference,
        spherely.symmetric_difference,
    ],
)
def test_concurrent_shared_inputs_match_serial(op, shared_pairs) -> None:
    dst, src = shared_pairs
    serial = _call_op(op, dst, src)

    for r in _fan_out(op, dst, src):
        np.testing.assert_array_equal(r, serial)


def test_concurrent_lazy_index_race() -> None:
    for _ in range(5):
        a = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
        b = spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))")
        dst = np.array([a] * 200, dtype=object)
        src = np.array([b] * 200, dtype=object)
        serial = spherely.area(spherely.intersection(dst, src), radius=1.0)

        a2 = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
        b2 = spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))")
        dst2 = np.array([a2] * 200, dtype=object)
        src2 = np.array([b2] * 200, dtype=object)
        for r in _fan_out(spherely.intersection, dst2, src2):
            np.testing.assert_array_equal(r, serial)


def test_concurrent_mixed_operations(shared_pairs) -> None:
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
            np.testing.assert_array_equal(got, expected[op])


def test_intersection_with_parallel_python_churn() -> None:
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


def test_refcounts_stable_after_concurrent_runs() -> None:
    a = spherely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
    b = spherely.from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))")
    dst = np.array([a] * 50, dtype=object)
    src = np.array([b] * 50, dtype=object)
    rc_a_before = sys.getrefcount(a)
    rc_b_before = sys.getrefcount(b)
    for _ in range(20):
        _fan_out(spherely.intersection, dst, src, n_threads=N_THREADS)
    gc.collect()
    assert sys.getrefcount(a) == rc_a_before
    assert sys.getrefcount(b) == rc_b_before
