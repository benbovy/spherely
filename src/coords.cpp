#include <s2/s2latlng.h>
#include <s2geography.h>

#include <iostream>
#include <stdexcept>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

double get_y(PyObjectGeography a) {
    auto geog = a.as_geog_ptr();
    if (geog->geog_type() != GeographyType::Point) {
        throw py::value_error("Only Point geometries supported");
    }
    const auto *point = geog->downcast_geog<s2geog::PointGeography>();
    auto s2point = point->Points()[0];
    auto latlng = S2LatLng(std::move(s2point));
    double lat = latlng.lat().degrees();
    return lat;
}

double get_x(PyObjectGeography a) {
    auto geog = a.as_geog_ptr();
    if (geog->geog_type() != GeographyType::Point) {
        throw py::value_error("Only Point geometries supported");
    }
    const auto *point = geog->downcast_geog<s2geog::PointGeography>();
    auto s2point = point->Points()[0];
    auto latlng = S2LatLng(std::move(s2point));
    double lng = latlng.lng().degrees();
    return lng;
}

void init_coords(py::module &m) {
    m.def("get_y",
          py::vectorize(&get_y),
          py::arg("a"),
          R"pbdoc(
        Returns the latitude value of the Point (in degrees).

        Parameters
        ----------
        a: :py:class:`Geography` or array_like
            Geography object(s).

    )pbdoc");

    m.def("get_x",
          py::vectorize(&get_x),
          py::arg("a"),
          R"pbdoc(
        Returns the longitude value of the Point (in degrees).

        Parameters
        ----------
        a: :py:class:`Geography` or array_like
            Geography object(s).

    )pbdoc");
}
