#include <s2geography.h>

#include "constants.hpp"
#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

PyObjectGeography centroid(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    auto s2_point = s2geog::s2_centroid(a_ptr);
    std::unique_ptr<Point> point =
        make_geography<s2geog::PointGeography, spherely::Point>(s2_point);
    return PyObjectGeography::from_geog(std::move(point));
}

PyObjectGeography boundary(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    auto s2_obj = s2geog::s2_boundary(a_ptr);
    // TODO return specific subclass
    auto geog_ptr = std::make_unique<spherely::Geography>(std::move(s2_obj));
    return PyObjectGeography::from_geog(std::move(geog_ptr));
}

PyObjectGeography convex_hull(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    auto s2_obj = s2geog::s2_convex_hull(a_ptr);
    auto geog_ptr = std::make_unique<spherely::Polygon>(std::move(s2_obj));
    return PyObjectGeography::from_geog(std::move(geog_ptr));
}

double distance(PyObjectGeography a, PyObjectGeography b, double radius = EARTH_RADIUS_METERS) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();
    return s2geog::s2_distance(a_index, b_index) * radius;
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
}
