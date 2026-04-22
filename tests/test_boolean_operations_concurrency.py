import os
import threading
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


def _fan_out(op, dst, src, n_threads=N_THREADS):
    """Run ``op(dst, src)`` from ``n_threads`` threads simultaneously."""
    barrier = threading.Barrier(n_threads)

    def work(_):
        barrier.wait()
        return _call_op(op, dst, src)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        return list(pool.map(work, range(n_threads)))


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
