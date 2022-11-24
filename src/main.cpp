#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


PYBIND11_MODULE(sksphere, m) {
    m.doc() = R"pbdoc(
        Scikit-Sphere
        -------------
        .. currentmodule:: sksphere
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
