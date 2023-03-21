#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void init_geography(py::module&);
void init_predicates(py::module&);
void init_accessors(py::module&);

PYBIND11_MODULE(spherely, m) {
    m.doc() = R"pbdoc(
        Spherely
        ---------
        .. currentmodule:: spherely
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    init_geography(m);
    init_predicates(m);
    init_accessors(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
