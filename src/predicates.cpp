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

void init_predicates(py::module& m) {
    m.def("intersects", py::vectorize(&intersects));
    m.def("equals", py::vectorize(&intersects));
}
