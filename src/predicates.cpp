#include <s2/s2boolean_operation.h>
#include <s2geography.h>

#include <functional>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

/*
** Functor for predicate bindings.
*/
class Predicate {
public:
    using FuncType = std::function<bool(const s2geog::ShapeIndexGeography&,
                                        const s2geog::ShapeIndexGeography&,
                                        const S2BooleanOperation::Options&)>;

    template <class F>
    Predicate(F&& func) : m_func(std::forward<F>(func)) {}

    template <class F>
    Predicate(F&& func, const S2BooleanOperation::Options& options)
        : m_func(std::forward<F>(func)), m_options(options) {}

    bool operator()(PyObjectGeography a, PyObjectGeography b) const {
        const auto& a_index = a.as_geog_ptr()->geog_index();
        const auto& b_index = b.as_geog_ptr()->geog_index();
        return m_func(a_index, b_index, m_options);
    }

private:
    FuncType m_func;
    S2BooleanOperation::Options m_options;
};

/*
** A specialization of the `Predicate` class for the touches operation,
** as two `S2BooleanOpteration::Options` objects are necessary.
*/
class TouchesPredicate {
public:
    TouchesPredicate() : m_closed_options(), m_open_options() {
        m_closed_options.set_polyline_model(S2BooleanOperation::PolylineModel::CLOSED);
        m_closed_options.set_polygon_model(S2BooleanOperation::PolygonModel::CLOSED);

        m_open_options.set_polyline_model(S2BooleanOperation::PolylineModel::OPEN);
        m_open_options.set_polygon_model(S2BooleanOperation::PolygonModel::OPEN);
    }

    bool operator()(PyObjectGeography a, PyObjectGeography b) const {
        const auto& a_index = a.as_geog_ptr()->geog_index();
        const auto& b_index = b.as_geog_ptr()->geog_index();

        return s2geog::s2_intersects(a_index, b_index, m_closed_options) &&
               !s2geog::s2_intersects(a_index, b_index, m_open_options);
    }

private:
    S2BooleanOperation::Options m_closed_options;
    S2BooleanOperation::Options m_open_options;
};

void init_predicates(py::module& m) {
    m.def("intersects",
          py::vectorize(Predicate(s2geog::s2_intersects)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(intersects(a, b)

        Returns True if A and B share any portion of space.

        Intersects implies that overlaps, touches and within are True.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

    )pbdoc");

    m.def("equals",
          py::vectorize(Predicate(s2geog::s2_equals)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(equals(a, b)

        Returns True if A and B are spatially equal.

        If A is within B and B is within A, A and B are considered equal. The
        ordering of points can be different.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

    )pbdoc");

    m.def("contains",
          py::vectorize(Predicate(s2geog::s2_contains)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(contains(a, b)

        Returns True if B is completely inside A.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

    )pbdoc");

    m.def("within",
          py::vectorize(Predicate([](const s2geog::ShapeIndexGeography& a_index,
                                     const s2geog::ShapeIndexGeography& b_index,
                                     const S2BooleanOperation::Options& options) {
              return s2geog::s2_contains(b_index, a_index, options);
          })),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(within(a, b)

        Returns True if A is completely inside B.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

    )pbdoc");

    m.def("disjoint",
          py::vectorize(Predicate([](const s2geog::ShapeIndexGeography& a_index,
                                     const s2geog::ShapeIndexGeography& b_index,
                                     const S2BooleanOperation::Options& options) {
              return !s2geog::s2_intersects(a_index, b_index, options);
          })),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(disjoint(a, b)

        Returns True if A boundaries and interior does not intersect at all
        with those of B.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

    )pbdoc");

    m.def("touches",
          py::vectorize(TouchesPredicate()),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(touches(a, b)

        Returns True if A and B intersect, but their interiors do not intersect.

        A and B must have at least one point in common, where the common point
        lies in at least one boundary.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

    )pbdoc");

    S2BooleanOperation::Options closed_options;
    closed_options.set_polyline_model(S2BooleanOperation::PolylineModel::CLOSED);
    closed_options.set_polygon_model(S2BooleanOperation::PolygonModel::CLOSED);

    m.def("covers",
          py::vectorize(Predicate(
              [](const s2geog::ShapeIndexGeography& a_index,
                 const s2geog::ShapeIndexGeography& b_index,
                 const S2BooleanOperation::Options& options) {
                  return s2geog::s2_contains(a_index, b_index, options);
              },
              closed_options)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(covers(a, b)

        Returns True if every point in B lies inside the interior or boundary of A.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

        Notes
        -----
        If A and B are both polygons and share co-linear edges,
        `covers` currently returns expected results only when those
        shared edges are identical.

    )pbdoc");

    m.def("covered_by",
          py::vectorize(Predicate(
              [](const s2geog::ShapeIndexGeography& a_index,
                 const s2geog::ShapeIndexGeography& b_index,
                 const S2BooleanOperation::Options& options) {
                  return s2geog::s2_contains(b_index, a_index, options);
              },
              closed_options)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(covered_by(a, b)

        Returns True if every point in A lies inside the interior or boundary of B.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bool or array

        See Also
        --------
        covers

    )pbdoc");
}
