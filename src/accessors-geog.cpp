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

double distance(PyObjectGeography a,
                PyObjectGeography b,
                double radius = numeric_constants::EARTH_RADIUS_METERS) {
    const auto& a_index = a.as_geog_ptr()->geog_index();
    const auto& b_index = b.as_geog_ptr()->geog_index();
    return s2geog::s2_distance(a_index, b_index) * radius;
}

double area(PyObjectGeography a, double radius = numeric_constants::EARTH_RADIUS_METERS) {
    return s2geog::s2_area(a.as_geog_ptr()->geog()) * radius * radius;
}

double length(PyObjectGeography a, double radius = numeric_constants::EARTH_RADIUS_METERS) {
    return s2geog::s2_length(a.as_geog_ptr()->geog()) * radius;
}

double perimeter(PyObjectGeography a, double radius = numeric_constants::EARTH_RADIUS_METERS) {
    return s2geog::s2_perimeter(a.as_geog_ptr()->geog()) * radius;
}

void init_accessors(py::module& m) {
    m.attr("EARTH_RADIUS_METERS") = py::float_(numeric_constants::EARTH_RADIUS_METERS);

    m.def("centroid",
          py::vectorize(&centroid),
          py::arg("geography"),
          py::pos_only(),
          R"pbdoc(centroid(geography, /)

        Computes the centroid of each geography.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        Geography or array
            A single or an array of POINT Geography object(s).

    )pbdoc");

    m.def("boundary",
          py::vectorize(&boundary),
          py::arg("geography"),
          py::pos_only(),
          R"pbdoc(boundary(geography, /)

        Computes the boundary of each geography.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        Geography or array
            A single or an array of either (MULTI)POINT or (MULTI)LINESTRING
            Geography object(s).

    )pbdoc");

    m.def("convex_hull",
          py::vectorize(&convex_hull),
          py::arg("geography"),
          py::pos_only(),
          R"pbdoc(convex_hull(geography, /)

        Computes the convex hull of each geography.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        Geography or array
            A single or an array of POLYGON Geography object(s).

    )pbdoc");

    m.def("distance",
          py::vectorize(&distance),
          py::arg("a"),
          py::arg("b"),
          py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS,
          R"pbdoc(distance(a, b, radius=spherely.EARTH_RADIUS_METERS)

        Calculate the distance between two geographies.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object(s).
        b : :py:class:`Geography` or array_like
            Geography object(s).
        radius : float, optional
            Radius of Earth in meters, default 6,371,010.

        Returns
        -------
        float or array
            Distance value(s), in meters.

    )pbdoc");

    m.def("area",
          py::vectorize(&area),
          py::arg("geography"),
          py::pos_only(),
          py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS,
          R"pbdoc(area(geography, /, radius=spherely.EARTH_RADIUS_METERS)

        Calculate the area of the geography.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).
        radius : float, optional
            Radius of Earth in meters, default 6,371,010.

        Returns
        -------
        float or array
            Area value(s), in square meters.

    )pbdoc");

    m.def("length",
          py::vectorize(&length),
          py::arg("geography"),
          py::pos_only(),
          py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS,
          R"pbdoc(length(geography, /, radius=spherely.EARTH_RADIUS_METERS)

        Calculates the length of a line geography, returning zero for other types.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).
        radius : float, optional
            Radius of Earth in meters, default 6,371,010.

        Returns
        -------
        float or array
            Length value(s), in meters.

   )pbdoc");

    m.def("perimeter",
          py::vectorize(&perimeter),
          py::arg("geography"),
          py::pos_only(),
          py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS,
          R"pbdoc(perimeter(geography, /, radius=spherely.EARTH_RADIUS_METERS)

        Calculates the perimeter of a polygon geography, returning zero for other types.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).
        radius : float, optional
            Radius of Earth in meters, default 6,371,010.

        Returns
        -------
        float or array
            Perimeter value(s), in meters.

    )pbdoc");
}
