#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void init_geography(py::module&);

PYBIND11_MODULE(s2shapely, m) {
    m.doc() = R"pbdoc(
        S2Shapely
        ---------
        .. currentmodule:: s2shapely
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    init_geography(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
