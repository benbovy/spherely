#include "s2/s2latlng.h"
#include "s2geography.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace s2g = s2geography;


using GeographyPtr = std::unique_ptr<s2g::Geography>;

/*
** Approch 1: internediate wrappers around s2geography objects
*/

// class PyGeography {
// public:
//     PyGeography(const PyGeography&) = delete;
//     PyGeography(PyGeography&& py_geog) = default;
//     PyGeography(GeographyPtr&& geog_ptr) : m_geog_ptr(std::move(geog_ptr)) {}

//     ~PyGeography() {
//         std::cout << "PyGeography destructor called" << this << std::endl;
//     }

//     PyGeography &operator=(const PyGeography&) = delete;
//     PyGeography &operator=(PyGeography&& other) = default;

//     int dimension() const { return m_geog_ptr->dimension(); }
//     int num_shapes() const { return m_geog_ptr->num_shapes(); }

//     GeographyPtr m_geog_ptr;
// };


// class PyPoint: public PyGeography {
// public:

//     PyPoint(GeographyPtr&& geog_ptr) : PyGeography(std::move(geog_ptr)) {};

//     static std::unique_ptr<PyPoint> FromLatLonDegrees(double lat_degrees, double lon_degrees) {
//         auto latlng = S2LatLng::FromDegrees(lat_degrees, lon_degrees);
//         GeographyPtr geog_ptr = std::make_unique<s2g::PointGeography>(S2Point(latlng));
//         std::unique_ptr<PyPoint> point_ptr = std::make_unique<PyPoint>(std::move(geog_ptr));
//         return point_ptr;
//     }
// };


/*
** Approach 2: Use s2geography types directly
*/

using PyGeography = s2g::Geography;
using PyPoint = s2g::PointGeography;


/*
** Factories
*/

class PointFactory {
public:

    static std::unique_ptr<PyPoint> FromLatLonDegrees(double lat_degrees, double lon_degrees) {
        auto latlng = S2LatLng::FromDegrees(lat_degrees, lon_degrees);
        return std::make_unique<PyPoint>(S2Point(latlng));
    }
};


/*
** Numpy funcs
*/

// from: https://github.com/pybind/pybind11/pull/1152
#define PYBIND11_NUMPY_OBJECT_DTYPE(Type)                                      \
  namespace pybind11 {                                                         \
  namespace detail {                                                           \
  template <> struct npy_format_descriptor<Type> {                             \
  public:                                                                      \
    enum { value = npy_api::NPY_OBJECT_ };                                     \
    static pybind11::dtype dtype() {                                           \
      if (auto ptr = npy_api::get().PyArray_DescrFromType_(value)) {           \
        return reinterpret_borrow<pybind11::dtype>(ptr);                       \
      }                                                                        \
      pybind11_fail("Unsupported buffer format!");                             \
    }                                                                          \
    static constexpr auto name = _("object");                                  \
  };                                                                           \
  }                                                                            \
  }

// make PyGeography pointer valid for use in numpy as object dtype
using PyGeographyPtr = PyGeography*;
PYBIND11_NUMPY_OBJECT_DTYPE(PyGeographyPtr);


PyGeographyPtr as_pygeography(py::object obj) {
    try {
        return obj.cast<PyGeographyPtr>();
    } catch (const py::cast_error &e) {
        throw py::value_error("not a Geography object");
    }
}


py::array_t<int> num_shapes(const py::array_t<PyGeographyPtr> geographies) {
    py::buffer_info buf = geographies.request();

    auto result = py::array_t<int>(buf.size);
    py::buffer_info result_buf = result.request();

    int *rptr = static_cast<int *>(result_buf.ptr);
    py::object *bptr = static_cast<py::object*>(buf.ptr);

    for(size_t i = 0; i < buf.size; i++) {
        // cast to a raw pointer here
        // pybind11's `type_caster<std::unique_ptr<wrapped_type>>`
        // doesnt't support Python -> C++ conversion (no `load` method)
        // as it would imply that Python needs to give up ownership of an object,
        // which is not possible (the object might be referenced elsewhere)
        auto geog_ptr = as_pygeography(bptr[i]);
        //rptr[i] = geog_ptr->m_geog_ptr->num_shapes();
        rptr[i] = geog_ptr->num_shapes();
    }

    return result;
}


py::array_t<PyGeographyPtr> create(py::array_t<double> xs, py::array_t<double> ys) {
  py::buffer_info xbuf = xs.request(), ybuf = ys.request();
  if (xbuf.ndim != 1 || ybuf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be one");
  }
  if (xbuf.size != ybuf.size) {
    throw std::runtime_error("Input shapes must match");
  }

  auto result = py::array_t<PyGeographyPtr>(xbuf.size);
  py::buffer_info rbuf = result.request();

  double *xptr = static_cast<double *>(xbuf.ptr);
  double *yptr = static_cast<double *>(ybuf.ptr);
  py::object *rptr = static_cast<py::object *>(rbuf.ptr);

  size_t size = static_cast<size_t>(xbuf.shape[0]);

  for (size_t i = 0; i < xbuf.shape[0]; i++) {
    auto point_ptr = PointFactory::FromLatLonDegrees(xptr[i], yptr[i]);
    // pybind11's `type_caster<std::unique_ptr<wrapped_type>>`
    // C++ -> Python (i.e., `cast`) move semantics
    rptr[i] = py::cast(std::move(point_ptr));
  }

  return result;
}



PYBIND11_MODULE(s2shapely, m) {
    m.doc() = R"pbdoc(
        S2Shapely
        ---------
        .. currentmodule:: s2shapely
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py::class_<PyGeography>(m, "Geography")
        .def_property_readonly("ndim", &PyGeography::dimension)
        .def_property_readonly("nshape", &PyGeography::num_shapes)
        .def("__repr__",
             [](const PyGeography &py_geog) {
                 s2g::WKTWriter writer;
                 //return writer.write_feature(*py_geog.m_geog_ptr);
                 return writer.write_feature(py_geog);
             }
        );

    py::class_<PyPoint, PyGeography>(m, "Point")
        .def(py::init(&PointFactory::FromLatLonDegrees));

    m.def("nshape", &num_shapes);
    m.def("create", &create);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
