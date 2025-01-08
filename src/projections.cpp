#include "projections.hpp"

#include <s2/s2projections.h>
#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
using namespace spherely;

void init_projections(py::module& m) {
    py::class_<Projection> projection(m, "Projection", R"pbdoc(
        Lightweight wrapper for selecting common reference systems used to
        project Geography points or vertices.

        Cannot be instantiated directly.

    )pbdoc");

    projection
        .def_static("lnglat", &Projection::lnglat, R"pbdoc(lnglat()

            Selects the "plate carree" projection.

            This projection maps coordinates on the sphere to (longitude, latitude) pairs.
            The x coordinates (longitude) span [-180, 180] and the y coordinates (latitude)
            span [-90, 90].

        )pbdoc")
        .def_static("pseudo_mercator", &Projection::pseudo_mercator, R"pbdoc(pseudo_mercator()

            Selects the spherical Mercator projection.

            When used together with WGS84 coordinates, known as the "Web
            Mercator" or "WGS84/Pseudo-Mercator" projection.

        )pbdoc")
        .def_static("orthographic",
                    &Projection::orthographic,
                    py::arg("longitude"),
                    py::arg("latitude"),
                    R"pbdoc(orthographic(longitude, latitude)

            Selects an orthographic projection with the given centre point.

            The resulting coordinates depict a single hemisphere of the globe as
            it appears from outer space, centred on the given point.

            Parameters
            ----------
            longitude : float
                Longitude coordinate of the center point, in degrees.
            latitude : float
                Latitude coordinate of the center point, in degrees.

        )pbdoc");
}
