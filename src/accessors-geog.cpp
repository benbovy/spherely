#include <s2geography.h>
#include <s2geography/geography.h>

#include "constants.hpp"
#include "creation.hpp"
#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

PyObjectGeography centroid(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    auto s2_point = s2geog::s2_centroid(a_ptr);
    return make_py_geography<s2geog::PointGeography>(s2_point);
}

PyObjectGeography boundary(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    return make_py_geography(s2geog::s2_boundary(a_ptr));
}

PyObjectGeography convex_hull(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    return make_py_geography(s2geog::s2_convex_hull(a_ptr));
}

double distance(PyObjectGeography a, PyObjectGeography b, double radius = EARTH_RADIUS_METERS) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();
    return s2geog::s2_distance(a_index, b_index) * radius;
}

double area(PyObjectGeography a, double radius = EARTH_RADIUS_METERS) {
    return s2geog::s2_area(a.as_geog_ptr()->geog()) * radius * radius;
}

double length(PyObjectGeography a, double radius = EARTH_RADIUS_METERS) {
    return s2geog::s2_length(a.as_geog_ptr()->geog()) * radius;
}

double perimeter(PyObjectGeography a, double radius = EARTH_RADIUS_METERS) {
    return s2geog::s2_perimeter(a.as_geog_ptr()->geog()) * radius;
}

void init_accessors(py::module& m) {
    m.attr("EARTH_RADIUS_METERS") = py::float_(EARTH_RADIUS_METERS);

    m.def("centroid",
          py::vectorize(&centroid),
          py::arg("a"),
          R"pbdoc(
        Computes the centroid of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");

    m.def("boundary",
          py::vectorize(&boundary),
          py::arg("a"),
          R"pbdoc(
        Computes the boundary of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");

    m.def("convex_hull",
          py::vectorize(&convex_hull),
          py::arg("a"),
          R"pbdoc(
        Computes the convex hull of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");

    m.def("distance",
          py::vectorize(&distance),
          py::arg("a"),
          py::arg("b"),
          py::arg("radius") = EARTH_RADIUS_METERS,
          R"pbdoc(
        Calculate the distance between two geographies.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object
        b : :py:class:`Geography` or array_like
            Geography object
        radius : float, optional
            Radius of Earth in meters, default 6,371,010

    )pbdoc");

    m.def("area",
          py::vectorize(&area),
          py::arg("a"),
          py::arg("radius") = EARTH_RADIUS_METERS,
          R"pbdoc(
        Calculate the area of the geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object
        radius : float, optional
            Radius of Earth in meters, default 6,371,010

    )pbdoc");

    m.def("length",
          py::vectorize(&length),
          py::arg("a"),
          py::arg("radius") = EARTH_RADIUS_METERS,
          R"pbdoc(
        Calculates the length of a line geography, returning zero for other types.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object
        radius : float, optional
            Radius of Earth in meters, default 6,371,010

   )pbdoc");

    m.def("perimeter",
          py::vectorize(&perimeter),
          py::arg("a"),
          py::arg("radius") = EARTH_RADIUS_METERS,
          R"pbdoc(
        Calculates the perimeter of a polygon geography, returning zero for other types.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object
        radius : float, optional
            Radius of Earth in meters, default 6,371,010
    )pbdoc");
}
