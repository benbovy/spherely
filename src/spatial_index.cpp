#include <pybind11/stl.h>
#include <s2/mutable_s2shape_index.h>
#include <s2/s2cell_id.h>
#include <s2/s2cell_union.h>
#include <s2/s2region_coverer.h>
#include <s2geography.h>
#include <s2geography/index.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "geography.hpp"
#include "predicates.hpp"
#include "pybind11.hpp"
#include "s2_tmp_memory_budget.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

namespace {

/*
** Sets s2geometry's process-wide index-build temporary memory budget
** (``FLAGS_s2shape_index_tmp_memory_budget``) for the lifetime of the guard and
** puts the previous value back when it goes out of scope, so an override cannot
** leak out of the build it was requested for.
**
** The flag is read and written through spherely::get/set_s2_tmp_memory_budget,
** which own the declaration it needs (see s2_tmp_memory_budget.cpp) -- reaching
** it directly from this file would not link against a shared s2 on Windows.
*/
class TmpMemoryBudgetGuard {
public:
    explicit TmpMemoryBudgetGuard(std::int64_t bytes) : m_previous(get_s2_tmp_memory_budget()) {
        set_s2_tmp_memory_budget(bytes);
    }

    ~TmpMemoryBudgetGuard() {
        set_s2_tmp_memory_budget(m_previous);
    }

    TmpMemoryBudgetGuard(const TmpMemoryBudgetGuard&) = delete;
    TmpMemoryBudgetGuard& operator=(const TmpMemoryBudgetGuard&) = delete;

private:
    std::int64_t m_previous;
};

}  // namespace

/*
** A spatial index over a collection of Geography objects, backed by
** s2geography::GeographyIndex (a MutableS2ShapeIndex). Provides fast candidate
** lookup similar to shapely's STRtree.
*/
class SpatialIndex {
public:
    SpatialIndex(const py::array_t<PyObjectGeography>& geographies) {
        if (geographies.ndim() != 1) {
            throw py::type_error("geographies must be a 1-dimensional array");
        }

        // store a shallow copy of the input array so that replacing its
        // elements afterwards cannot drop Geography objects whose S2Shapes
        // are still borrowed by the index
        m_geographies = py::array_t<PyObjectGeography>::ensure(geographies.attr("copy")());
        m_index = std::make_unique<s2geog::GeographyIndex>();

        auto n = m_geographies.size();
        auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
        for (py::ssize_t i = 0; i < n; i++) {
            m_index->Add(data[i].as_geog_ptr()->geog(), static_cast<int>(i));
        }
    }

    std::size_t size() const {
        return static_cast<std::size_t>(m_geographies.size());
    }

    // True once the queued updates have been applied (no build work pending).
    bool is_built() const {
        return m_index->ShapeIndex().is_fresh();
    }

    // Apply the updates queued by the constructor, optionally under a raised
    // s2 temporary memory budget. A no-op once the index is built.
    void build(std::optional<std::int64_t> tmp_memory_budget) const {
        if (tmp_memory_budget.has_value() && *tmp_memory_budget <= 0) {
            throw py::value_error("tmp_memory_budget must be a positive number of bytes, got " +
                                  std::to_string(*tmp_memory_budget) +
                                  " (s2geometry's default budget is 104857600 bytes)");
        }

        // nothing pending: no build to tune, so leave the process-wide flag alone
        if (is_built()) {
            return;
        }

        const auto& shape_index = m_index->ShapeIndex();
        if (tmp_memory_budget.has_value()) {
            TmpMemoryBudgetGuard guard(*tmp_memory_budget);
            shape_index.ForceBuild();
        } else {
            shape_index.ForceBuild();
        }
    }

    py::array geometries() const {
        return m_geographies;
    }

    // Scalar query: single Geography -> 1-d array of sorted tree indices.
    py::array_t<py::ssize_t> query(const Geography& geography,
                                   std::optional<std::string> predicate) const {
        auto pred = make_predicate(predicate);
        auto results = query_one(geography, pred ? &*pred : nullptr);
        return to_index_array(results);
    }

    // Array query: 1-d array of Geography -> (2, K) array of
    // (input index, tree index) pairs.
    py::array_t<py::ssize_t> query(const py::array_t<PyObjectGeography>& geographies,
                                   std::optional<std::string> predicate) const {
        if (geographies.ndim() != 1) {
            throw py::type_error(
                "query geography must be a Geography or a 1-dimensional array of Geography");
        }

        auto pred = make_predicate(predicate);

        auto n = geographies.size();
        auto* data = static_cast<PyObjectGeography*>(geographies.request().ptr);
        std::vector<py::ssize_t> input_idx;
        std::vector<py::ssize_t> tree_idx;
        for (py::ssize_t i = 0; i < n; i++) {
            auto results = query_one(*data[i].as_geog_ptr(), pred ? &*pred : nullptr);
            for (int t : results) {
                input_idx.push_back(i);
                tree_idx.push_back(static_cast<py::ssize_t>(t));
            }
        }

        auto k = input_idx.size();
        py::array_t<py::ssize_t> out({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(k)});
        auto out_data = out.mutable_unchecked<2>();
        for (std::size_t j = 0; j < k; j++) {
            out_data(0, static_cast<py::ssize_t>(j)) = input_idx[j];
            out_data(1, static_cast<py::ssize_t>(j)) = tree_idx[j];
        }
        return out;
    }

private:
    std::unique_ptr<s2geog::GeographyIndex> m_index;
    // also keeps the Geography objects alive (the index borrows their S2Shapes)
    py::array_t<PyObjectGeography> m_geographies;

    static std::optional<PredicateFunc> make_predicate(const std::optional<std::string>& name) {
        if (!name.has_value()) {
            return std::nullopt;
        }
        return get_predicate(*name);
    }

    // Return the sorted tree indices whose cells overlap the query geography,
    // optionally refined by ``pred`` (predicate(query, candidate)).
    std::vector<int> query_one(const Geography& query_geog, const PredicateFunc* pred) const {
        std::unordered_set<int> hits;

        auto region = query_geog.geog().Region();
        S2RegionCoverer coverer;
        std::vector<S2CellId> covering;
        coverer.GetCovering(*region, &covering);

        s2geog::GeographyIndex::Iterator iter(m_index.get());
        iter.Query(covering, &hits);

        std::vector<int> results;
        if (pred == nullptr) {
            results.assign(hits.begin(), hits.end());
        } else {
            const auto& query_index = query_geog.geog_index();
            auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
            for (int candidate : hits) {
                auto* cand_geog = data[candidate].as_geog_ptr();
                if ((*pred)(query_index, cand_geog->geog_index())) {
                    results.push_back(candidate);
                }
            }
        }

        std::sort(results.begin(), results.end());
        return results;
    }

    static py::array_t<py::ssize_t> to_index_array(const std::vector<int>& indices) {
        auto n = indices.size();
        py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(n));
        auto out_data = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < n; i++) {
            out_data(static_cast<py::ssize_t>(i)) = static_cast<py::ssize_t>(indices[i]);
        }
        return out;
    }
};

void init_spatial_index(py::module& m) {
    py::class_<SpatialIndex>(m, "SpatialIndex", R"pbdoc(
        A spatial index for a collection of geographies.

        The index allows fast retrieval of the geographies that may interact
        with a given geography, e.g., for accelerating spatial joins. It is
        built on top of s2geometry's shape index and is conceptually similar to
        shapely's ``STRtree``.

        The underlying shape index is built lazily: the constructor only queues
        the geographies and the first query pays for the build. Use
        :py:meth:`SpatialIndex.build` to do that work up front instead.

        Parameters
        ----------
        geographies : array_like
            An array (or other sequence) of :py:class:`Geography` objects. Empty
            geographies are indexed but never returned by queries.

    )pbdoc")
        .def(py::init<const py::array_t<PyObjectGeography>&>(),
             py::arg("geographies"),
             "__init__(self, geographies)")
        .def("build",
             &SpatialIndex::build,
             py::arg("tmp_memory_budget") = py::none(),
             R"pbdoc(build(tmp_memory_budget=None)

        Force the deferred index build, optionally under a raised temporary
        memory budget.

        The index is built lazily, so without this call the first query pays
        the whole build cost. Calling it is optional, never changes the result
        of a query made with a ``predicate``, and does nothing once the index
        is built (see :py:attr:`SpatialIndex.is_built`).

        Parameters
        ----------
        tmp_memory_budget : int, optional
            Temporary memory budget for this build, in bytes. s2geometry builds
            the index in batches sized to fit it, and many batches cost more
            than one, so the knob is only useful *raised*; about 226 bytes per
            edge is enough to build in a single batch. Must be strictly
            positive, and must fit in a signed 64-bit integer (larger values
            raise ``TypeError``). The previous value is put back when the call
            returns, so the override does not leak into any later build.

            ``None`` does *not* mean "no budget": it leaves s2geometry's
            setting untouched (104857600 bytes, i.e. 100 MB, unless something
            else in the process changed it), so the build still batches.

        Notes
        -----
        Peak resident memory can rise by roughly the budget granted, and
        nothing is granted unless a budget is passed.

        The budget changes where s2geometry places its batch boundaries, and so
        how it subdivides the index. A query made without a ``predicate`` can
        therefore return a different candidate set under a different budget --
        candidates may be added *or* dropped, and the sets for two budgets need
        not be nested. What does not change is that the candidate set is always
        a superset of the true matches, so any ``predicate`` refines it to the
        same answer.

        This call holds the GIL for its whole duration -- minutes, for a large
        collection -- and the budget is a process-wide s2geometry setting for
        that duration.

        Examples
        --------
        >>> import spherely
        >>> index = spherely.SpatialIndex(geographies)
        >>> index.build(tmp_memory_budget=4 * 1024**3)  # 4 GiB

    )pbdoc")
        .def("__len__", &SpatialIndex::size)
        .def_property_readonly("is_built",
                               &SpatialIndex::is_built,
                               R"pbdoc(Whether the underlying shape index has been built.

        ``False`` between the constructor and the first
        :py:meth:`SpatialIndex.build` or query, ``True`` afterwards. An index
        over an empty collection is ``True`` from the start.

    )pbdoc")
        .def_property_readonly("geometries",
                               &SpatialIndex::geometries,
                               "The array of geographies in the index (in input order).")
        .def("query",
             py::overload_cast<const Geography&, std::optional<std::string>>(&SpatialIndex::query,
                                                                             py::const_),
             py::arg("geography"),
             py::arg("predicate") = py::none(),
             R"pbdoc(query(geography, predicate=None)

        Return the integer indices of all geographies in the index whose cells
        overlap the given geography (a coarse candidate set), optionally refined
        by a spatial predicate.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            The geography or geographies to query.
        predicate : str, optional
            If provided, only return candidates for which
            ``predicate(geography, indexed_geography)`` is True. One of
            "intersects", "within", "contains", "covers", "covered_by",
            "touches" or "equals".

        Returns
        -------
        ndarray
            If ``geography`` is a scalar, a 1-d array of (sorted) indices into
            the index. If ``geography`` is an array, a ``(2, N)`` array where the
            first row holds the input geography indices and the second row the
            matching index (tree) indices.

    )pbdoc")
        .def("query",
             py::overload_cast<const py::array_t<PyObjectGeography>&, std::optional<std::string>>(
                 &SpatialIndex::query, py::const_),
             py::arg("geography"),
             py::arg("predicate") = py::none());

    // Not public API: exposed so that the test suite can move the process-wide
    // budget off its default and assert what SpatialIndex.build() puts back.
    m.def("_s2_tmp_memory_budget",
          &get_s2_tmp_memory_budget,
          "Return s2geometry's current index-build temporary memory budget, in bytes.");
    m.def("_set_s2_tmp_memory_budget",
          &set_s2_tmp_memory_budget,
          py::arg("bytes"),
          "Set s2geometry's index-build temporary memory budget, in bytes.");
}
