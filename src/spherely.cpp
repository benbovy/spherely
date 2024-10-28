#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void init_geography(py::module&);
void init_creation(py::module&);
void init_predicates(py::module&);
void init_accessors(py::module&);
void init_io(py::module&);

PYBIND11_MODULE(spherely, m) {
    m.doc() = R"pbdoc(
        Spherely
        ---------
        .. currentmodule:: spherely
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    init_geography(m);
    init_creation(m);
    init_predicates(m);
    init_accessors(m);
    init_io(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

#ifdef S2GEOGRAPHY_VERSION
    m.attr("__s2geography_version__") = MACRO_STRINGIFY(S2GEOGRAPHY_VERSION);
#endif
}
