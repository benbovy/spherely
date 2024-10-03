#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

PyObjectGeography from_wkt(py::str a) {
    s2geog::WKTReader reader;
    std::unique_ptr<s2geog::Geography> s2geog = reader.read_feature(a);
    auto geog_ptr = std::make_unique<spherely::Geography>(std::move(s2geog));
    return PyObjectGeography::from_geog(std::move(geog_ptr));
}

py::str to_wkt(PyObjectGeography a) {
    s2geog::WKTWriter writer;
    auto res = writer.write_feature(a.as_geog_ptr()->geog());
    return py::str(res);
}

void init_io(py::module& m) {
    m.def("from_wkt",
          py::vectorize(&from_wkt),
          py::arg("a"),
          R"pbdoc(
        Creates geographies from the Well-Known Text (WKT) representation.

        Parameters
        ----------
        a : str or array_like
            WKT strings.

    )pbdoc");

    m.def("to_wkt",
          py::vectorize(&to_wkt),
          py::arg("a"),
          R"pbdoc(
        Returns the WKT representation of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");
}
