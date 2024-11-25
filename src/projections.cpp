#include "projections.hpp"

#include <s2/s2projections.h>
#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

void init_projections(py::module& m) {
    py::class_<Projection>(m, "Projection")
        .def("lnglat", &Projection::lnglat)
        .def("pseudo_mercator", &Projection::pseudo_mercator)
        .def("orthographic", &Projection::orthographic);
}
