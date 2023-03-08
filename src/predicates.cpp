#include <s2/s2boolean_operation.h>
#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

bool intersects(PyObjectGeography a, PyObjectGeography b) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();

    S2BooleanOperation::Options options;
    return s2geog::s2_intersects(a_index, b_index, options);
}

bool equals(PyObjectGeography a, PyObjectGeography b) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();

    S2BooleanOperation::Options options;
    return s2geog::s2_equals(a_index, b_index, options);
}

bool contains(PyObjectGeography a, PyObjectGeography b) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();

    S2BooleanOperation::Options options;
    return s2geog::s2_contains(a_index, b_index, options);
}

bool within(PyObjectGeography a, PyObjectGeography b) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();

    S2BooleanOperation::Options options;
    return s2geog::s2_contains(b_index, a_index, options);
}

bool disjoint(PyObjectGeography a, PyObjectGeography b) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();

    S2BooleanOperation::Options options;
    return !s2geog::s2_intersects(a_index, b_index, options);
}

void init_predicates(py::module& m) {
    m.def("intersects", py::vectorize(&intersects), py::arg("a"), py::arg("b"),
          R"pbdoc(
        Returns True if A and B share any portion of space.

        Intersects implies that overlaps, touches and within are True.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    m.def("equals", py::vectorize(&equals), py::arg("a"), py::arg("b"),
          R"pbdoc(
        Returns True if A and B are spatially equal.

        If A is within B and B is within A, A and B are considered equal. The
        ordering of points can be different.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    m.def("contains", py::vectorize(&contains), py::arg("a"), py::arg("b"),
          R"pbdoc(

        Returns True if B is completely inside A.
        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");
    
    m.def("within", py::vectorize(&within), py::arg("a"), py::arg("b"),
          R"pbdoc(
        Returns True if A is completely inside B.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    m.def("disjoint", py::vectorize(&disjoint), py::arg("a"), py::arg("b"),
          R"pbdoc(
        Returns True if A boundaries and interior does not intersect at all
        with those of B.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    }