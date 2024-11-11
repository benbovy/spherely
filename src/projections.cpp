#include "projections.hpp"

#include <s2/s2projections.h>
#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

// std::tuple<double, double> project_mercator(py::object obj) {
//     auto geog = (static_cast<PyObjectGeography&>(obj)).as_geog_ptr();
//     if (geog->geog_type() != GeographyType::Point) {
//         throw py::value_error("test");
//     }
//     auto point = static_cast<Point*>(geog);
//     auto s2point = point->s2point();

//     auto projection = S2::MercatorProjection(20037508.3427892);
//     auto r2point = projection.Project(s2point);
//     double x = r2point.x();
//     double y = r2point.y();

//     return std::make_tuple(x, y);
// }

void init_projections(py::module& m) {
    // m.def("project_mercator", &project_mercator);
    py::class_<Projection>(m, "Projection")
        .def("lnglat", &Projection::lnglat)
        .def("pseudo_mercator", &Projection::pseudo_mercator)
        .def("orthographic", &Projection::orthographic);
}
