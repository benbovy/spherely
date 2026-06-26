#include <pybind11/stl.h>
#include <s2/s2cell_id.h>
#include <s2/s2cell_union.h>
#include <s2/s2region_coverer.h>
#include <s2geography.h>
#include <s2geography/index.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "geography.hpp"
#include "predicates.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

/*
** A spatial index over a collection of Geography objects, backed by
** s2geography::GeographyIndex (a MutableS2ShapeIndex). Provides fast candidate
** lookup similar to shapely's STRtree.
**
** The input geographies are held as a numpy object-dtype array, which both
** keeps the underlying C++ Geography objects (whose S2Shapes are borrowed by
** the index) alive and backs the ``geometries`` property.
*/
class SpatialIndex {
public:
    SpatialIndex(py::object geographies) {
        auto arr = py::array_t<PyObjectGeography>::ensure(geographies);
        if (!arr) {
            throw py::type_error("geographies must be an array-like of Geography objects");
        }
        if (arr.ndim() != 1) {
            throw py::type_error("geographies must be a 1-dimensional array");
        }

        m_geographies = arr;
        m_index = std::make_unique<s2geog::GeographyIndex>();

        auto n = arr.size();
        auto* data = static_cast<PyObjectGeography*>(arr.request().ptr);
        for (py::ssize_t i = 0; i < n; i++) {
            auto* geog_ptr = data[i].as_geog_ptr();
            m_index->Add(geog_ptr->geog(), static_cast<int>(i));
        }
    }

    std::size_t size() const {
        return static_cast<std::size_t>(m_geographies.size());
    }

    py::array geometries() const {
        return m_geographies;
    }

    // Dispatch between scalar (single Geography -> 1-d index array) and
    // array-like (-> (2, K) array of (input index, tree index) pairs) queries.
    py::object query(py::object geography, std::optional<std::string> predicate) const {
        PredicateFunc pred;
        bool has_pred = predicate.has_value();
        if (has_pred) {
            pred = get_predicate(*predicate);
        }

        // PyObjectGeography is layout-compatible with py::object (see
        // PyObjectGeography::from_geog), so a reference cast is safe here.
        auto& maybe_geog = static_cast<PyObjectGeography&>(geography);
        if (maybe_geog.is_geog_ptr()) {
            auto results = query_one(maybe_geog.as_geog_ptr(), has_pred ? &pred : nullptr);
            return to_index_array(results);
        }

        auto arr = py::array_t<PyObjectGeography>::ensure(geography);
        if (!arr || arr.ndim() != 1) {
            throw py::type_error(
                "query geography must be a Geography or a 1-dimensional array of Geography");
        }

        auto n = arr.size();
        auto* data = static_cast<PyObjectGeography*>(arr.request().ptr);
        std::vector<py::ssize_t> input_idx;
        std::vector<py::ssize_t> tree_idx;
        for (py::ssize_t i = 0; i < n; i++) {
            auto results = query_one(data[i].as_geog_ptr(), has_pred ? &pred : nullptr);
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
    py::array_t<PyObjectGeography> m_geographies;

    // Return the sorted tree indices whose cells overlap the query geography,
    // optionally refined by ``pred`` (predicate(query, candidate)).
    std::vector<int> query_one(Geography* query_geog, const PredicateFunc* pred) const {
        std::unordered_set<int> hits;

        auto region = query_geog->geog().Region();
        S2RegionCoverer coverer;
        std::vector<S2CellId> covering;
        coverer.GetCovering(*region, &covering);

        s2geog::GeographyIndex::Iterator iter(m_index.get());
        iter.Query(covering, &hits);

        std::vector<int> results;
        if (pred == nullptr) {
            results.assign(hits.begin(), hits.end());
        } else {
            const auto& query_index = query_geog->geog_index();
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

        Parameters
        ----------
        geographies : array_like
            An array (or other sequence) of :py:class:`Geography` objects. Empty
            geographies are indexed but never returned by queries.

    )pbdoc")
        .def(py::init<py::object>(), py::arg("geographies"))
        .def("__len__", &SpatialIndex::size)
        .def_property_readonly("geometries",
                               &SpatialIndex::geometries,
                               "The array of geographies in the index (in input order).")
        .def("query",
             &SpatialIndex::query,
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

    )pbdoc");
}
