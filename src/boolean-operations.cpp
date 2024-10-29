#include <s2geography.h>
#include <s2geography/geography.h>

#include "constants.hpp"
#include "creation.hpp"
#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

class BooleanOp {
public:
    BooleanOp(S2BooleanOperation::OpType op_type)
        : m_op_type(op_type), m_options(s2geog::GlobalOptions()) {
        // TODO make this configurable
        // m_options.polyline_layer_action = s2geography::GlobalOptions::OUTPUT_ACTION_IGNORE;
    }

    PyObjectGeography operator()(PyObjectGeography a, PyObjectGeography b) const {
        const auto& a_index = a.as_geog_ptr()->geog_index();
        const auto& b_index = b.as_geog_ptr()->geog_index();
        std::unique_ptr<s2geog::Geography> geog_out =
            s2geog::s2_boolean_operation(a_index, b_index, m_op_type, m_options);

        return make_py_geography(std::move(geog_out));
    }

private:
    S2BooleanOperation::OpType m_op_type;
    s2geog::GlobalOptions m_options;
};

// PyObjectGeography intersection(PyObjectGeography a, PyObjectGeography b) {
//     const auto& a_index = a.as_geog_ptr()->geog_index();
//     const auto& b_index = b.as_geog_ptr()->geog_index();

//     std::unique_ptr<s2geog::Geography> geog_out = s2geog::s2_boolean_operation(
//       a_index, b_index, S2BooleanOperation::OpType::INTERSECTION,
//       s2geog::GlobalOptions());

//     return make_py_geography(std::move(geog_out));
// }

void init_boolean_operations(py::module& m) {
    m.def("union",
          py::vectorize(BooleanOp(S2BooleanOperation::OpType::UNION)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Computes the union of both geographies.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");

    m.def("intersection",
          py::vectorize(BooleanOp(S2BooleanOperation::OpType::INTERSECTION)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Computes the intersection of both geographies.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");

    m.def("difference",
          py::vectorize(BooleanOp(S2BooleanOperation::OpType::DIFFERENCE)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Computes the difference of both geographies.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");

    m.def("symmetric_difference",
          py::vectorize(BooleanOp(S2BooleanOperation::OpType::SYMMETRIC_DIFFERENCE)),
          py::arg("a"),
          py::arg("b"),
          R"pbdoc(
        Computes the difference of both geographies.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");
}
