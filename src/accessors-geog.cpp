#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

PyObjectGeography convex_hull(PyObjectGeography a) {
    const auto& a_ptr = a.as_geog_ptr()->geog();
    auto res = s2geog::s2_convex_hull(a_ptr);
    auto res_geog = spherely::Geography(std::move(res));
    auto res_geog_unique = std::make_unique<spherely::Geography>(std::move(res_geog));
    auto res_object = PyObjectGeography::as_py_object(std::move(res_geog_unique));
    return static_cast<PyObjectGeography&>(res_object);
}

void init_accessors(py::module& m) {
    m.def("convex_hull", py::vectorize(&convex_hull), py::arg("a"),
          R"pbdoc(
        Returns True if A and B share any portion of space.

        Intersects implies that overlaps, touches and within are True.

        Parameters
        ----------
        a, b : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");
}
