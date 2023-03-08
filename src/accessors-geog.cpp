#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

PyObjectGeography convex_hull(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    auto s2_obj = s2geog::s2_convex_hull(a_ptr);
    auto geog_ptr = std::make_unique<Geography>(Geography(std::move(s2_obj)));
    auto res_object = PyObjectGeography::as_py_object(std::move(geog_ptr));
    return static_cast<PyObjectGeography&>(res_object);
}

void init_accessors(py::module& m) {
    m.def("convex_hull", py::vectorize(&convex_hull), py::arg("a"),
          R"pbdoc(
        Computes the convex hull of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object

    )pbdoc");
}
