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

    bool operator()(PyObjectGeography a, PyObjectGeography b) const {
        const auto& a_index = a.as_geog_ptr()->geog_index();
        const auto& b_index = b.as_geog_ptr()->geog_index();
        return m_func(a_index, b_index, m_options);
    }

private:
    FuncType m_func;
    S2BooleanOperation::Options m_options;
};

void init_predicates(py::module& m) {
    m.def("intersects",
          py::vectorize(Predicate(s2geog::s2_intersects)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Returns True if A and B share any portion of space.

        Intersects implies that overlaps, touches and within are True.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    m.def("equals",
          py::vectorize(Predicate(s2geog::s2_equals)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Returns True if A and B are spatially equal.

        If A is within B and B is within A, A and B are considered equal. The
        ordering of points can be different.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    m.def("contains",
          py::vectorize(Predicate(s2geog::s2_contains)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Returns True if B is completely inside A.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    m.def("within",
          py::vectorize(Predicate([](const s2geog::ShapeIndexGeography& a_index,
                                     const s2geog::ShapeIndexGeography& b_index,
                                     const S2BooleanOperation::Options& options) {
              return s2geog::s2_contains(b_index, a_index, options);
          })),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Returns True if A is completely inside B.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    m.def("disjoint",
          py::vectorize(Predicate([](const s2geog::ShapeIndexGeography& a_index,
                                     const s2geog::ShapeIndexGeography& b_index,
                                     const S2BooleanOperation::Options& options) {
              return !s2geog::s2_intersects(a_index, b_index, options);
          })),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Returns True if A boundaries and interior does not intersect at all
        with those of B.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");
}
