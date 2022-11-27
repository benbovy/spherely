#include "s2/s2latlng.h"
#include "s2geography.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace s2geog = s2geography;


using GeographyPtr = std::unique_ptr<s2geog::Geography>;

/*
** Approach 1: internediate wrappers around s2geography objects with move semantics
** (useful for debugging but probably not needed)
*/

class Geography {
public:
    Geography(const Geography&) = delete;
    Geography(Geography&& py_geog) : m_geog_ptr(std::move(py_geog.m_geog_ptr)) {
        std::cout << "Geography move constructor called: " << this << std::endl;
    }
    Geography(GeographyPtr&& geog_ptr) : m_geog_ptr(std::move(geog_ptr)) {}

    ~Geography() {
        std::cout << "Geography destructor called: " << this << std::endl;
    }

    Geography& operator=(const Geography&) = delete;
    Geography& operator=(Geography&& other) {
        std::cout << "Geography move assignment called: " << this << std::endl;
        m_geog_ptr = std::move(other.m_geog_ptr);
        return *this;
    }

    int dimension() const { return m_geog_ptr->dimension(); }
    int num_shapes() const { return m_geog_ptr->num_shapes(); }

    GeographyPtr m_geog_ptr;
};


class Point: public Geography {
public:

    Point(GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

};


class LineString: public Geography {
public:

    LineString(GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

};


/*
** Approach 2: Use s2geography types directly
** (type aliases used for convenience)
*/

//using Geography = s2geog::Geography;
//using Point = s2geog::PointGeography;
//using LineString = s2geog::PolylineGeography;


/*
** Factories
*/

class PointFactory {
public:

    // static std::unique_ptr<Point> FromLatLonDegrees(double lat_degrees, double lon_degrees) {
    //     auto latlng = S2LatLng::FromDegrees(lat_degrees, lon_degrees);
    //     return std::make_unique<Point>(S2Point(latlng));
    // }

    static std::unique_ptr<Point> FromLatLonDegrees(double lat_degrees, double lon_degrees) {
        auto latlng = S2LatLng::FromDegrees(lat_degrees, lon_degrees);
        GeographyPtr geog_ptr = std::make_unique<s2geog::PointGeography>(S2Point(latlng));
        std::unique_ptr<Point> point_ptr = std::make_unique<Point>(std::move(geog_ptr));
        return point_ptr;
    }
};


class LineStringFactory {
public:
    using LatLonCoords = std::vector<std::pair<double, double>>;

    static std::unique_ptr<LineString> FromLatLonCoords(LatLonCoords coords) {
        std::vector<S2LatLng> latlng_pts;
        for (auto& latlng : coords) {
            latlng_pts.push_back(S2LatLng::FromDegrees(latlng.first, latlng.second));
        }
        auto polyline = std::make_unique<S2Polyline>(latlng_pts);
        GeographyPtr geog_ptr = std::make_unique<s2geog::PolylineGeography>(std::move(polyline));
        return std::make_unique<LineString>(std::move(geog_ptr));
    }
};


/*
** Pybind11 patches and workarounds for operating on Python wrapped Geography
** objects through Numpy arrays and universal functions (using numpy.object
** dtype).
**
** Somewhat hacky!
*/

// A ``pybind11::object`` that maybe points to a ``Geography`` C++ object.
//
// To main goal of this class is to be used as argument type of s2shapely's
// vectorized functions that operate on Geography objects via the numpy.object
// dtype.
//
// Instead of relying on Pybind11's implicit conversion mechanisms (copy), we
// require explicit conversion from / to ``pybind11::object``.
//
class PyObjectGeography : public py::object {
public:

    // Python -> C++ conversion
    //
    // Raises a ``TypeError`` on the Python side if the cast fails.
    //
    // Note: a raw pointer is used here because Pybind11's
    // `type_caster<std::unique_ptr<wrapped_type>>` doesnt't support Python->C++
    // conversion (no `load` method) as it would imply that Python needs to give
    // up ownership of an object, which is not possible (the object might be
    // referenced elsewhere)
    //
    // Conversion shouldn't involve any copy. The cast is dynamic, though, as
    // needed since numpy.object dtype can refer to any Python object.
    //
    Geography* as_geog_ptr() const {
        try {
            return cast<Geography*>();
        } catch (const py::cast_error &e) {
            throw py::type_error("not a Geography object");
        }
    }

    // C++ -> Python conversion
    //
    // Note: pybind11's `type_caster<std::unique_ptr<wrapped_type>>` implements
    // move semantics (Python takes ownership).
    //
    template <class T, std::enable_if_t<std::is_base_of<Geography, T>::value, bool> = true>
    static py::object as_py_object(std::unique_ptr<T> geog_ptr) {
        return py::cast(std::move(geog_ptr));
    }
};

namespace pybind11 {
    namespace detail {

        // Force pybind11 to allow PyObjectGeography as argument of vectorized
        // functions.
        //
        // Pybind11 doesn't support non-POD types as arguments for vectorized
        // functions because of its internal conversion mechanisms and also
        // because direct memory access requires a standard layout type.
        //
        // Here it is probably fine to make an exception since we require
        // explicit Python object <-> C++ Geography conversion and also because
        // with the numpy.object dtype the data are actually references to Python
        // objects (not the objects themselves).
        //
        // Caveat: be careful and use PyObjectGeography cast methods!
        //
        template <>
        struct vectorize_arg<PyObjectGeography> {
            using T = PyObjectGeography;
            // The wrapped function gets called with this type:
            using call_type = T;
            // Is this a vectorized argument?
            static constexpr bool vectorize = true;
            // Accept this type: an array for vectorized types, otherwise the type as-is:
            using type = conditional_t<vectorize, array_t<remove_cv_t<call_type>, array::forcecast>, T>;
        };

        // Register PyObjectGeography as a valid numpy dtype (numpy.object alias)
        // from: https://github.com/pybind/pybind11/pull/1152
        template <>
        struct npy_format_descriptor<PyObjectGeography> {
            static constexpr auto name = _("object");
            enum { value = npy_api::NPY_OBJECT_ };
            static pybind11::dtype dtype()
            {
                if (auto ptr = npy_api::get().PyArray_DescrFromType_(value))
                {
                    return reinterpret_borrow<pybind11::dtype>(ptr);
                }
                pybind11_fail("Unsupported buffer format!");
            }
        };
    }
}


/*
** Test Numpy-vecotized API
*/

py::array_t<int> num_shapes(const py::array_t<PyObjectGeography> geographies) {
    py::buffer_info buf = geographies.request();

    auto result = py::array_t<int>(buf.size);
    py::buffer_info result_buf = result.request();
    int *rptr = static_cast<int *>(result_buf.ptr);

    for(size_t i = 0; i < buf.size; i++) {
        auto geog_ptr = (*geographies.data(i)).as_geog_ptr();
        rptr[i] = geog_ptr->num_shapes();
        //std::cout << sizeof(*geographies.data(i)) << " - " << sizeof(geog_ptr) << std::endl;
    }

    return result;
}


py::array_t<PyObjectGeography> create(py::array_t<double> xs, py::array_t<double> ys) {
  py::buffer_info xbuf = xs.request(), ybuf = ys.request();
  if (xbuf.ndim != 1 || ybuf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be one");
  }
  if (xbuf.size != ybuf.size) {
    throw std::runtime_error("Input shapes must match");
  }

  auto result = py::array_t<PyObjectGeography>(xbuf.size);
  py::buffer_info rbuf = result.request();

  double *xptr = static_cast<double *>(xbuf.ptr);
  double *yptr = static_cast<double *>(ybuf.ptr);
  py::object *rptr = static_cast<py::object *>(rbuf.ptr);

  size_t size = static_cast<size_t>(xbuf.shape[0]);

  for (size_t i = 0; i < xbuf.shape[0]; i++) {
    auto point_ptr = PointFactory::FromLatLonDegrees(xptr[i], yptr[i]);
    rptr[i] = PyObjectGeography::as_py_object(std::move(point_ptr));
  }

  return result;
}


int get_dimensions(PyObjectGeography obj) {
    return obj.as_geog_ptr()->dimension();
}


PYBIND11_MODULE(s2shapely, m) {
    m.doc() = R"pbdoc(
        S2Shapely
        ---------
        .. currentmodule:: s2shapely
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py::class_<Geography>(m, "Geography")
        .def_property_readonly("dimensions", &Geography::dimension)
        .def_property_readonly("nshape", &Geography::num_shapes)
        .def("__repr__",
             [](const Geography &py_geog) {
                 s2geog::WKTWriter writer;
                 return writer.write_feature(*py_geog.m_geog_ptr);
                 //return writer.write_feature(py_geog);
             }
        );

    py::class_<Point, Geography>(m, "Point")
        .def(py::init(&PointFactory::FromLatLonDegrees));

    py::class_<LineString, Geography>(m, "LineString")
        .def(py::init(&LineStringFactory::FromLatLonCoords));

    m.def("nshape", &num_shapes);
    m.def("create", &create);
    m.def("get_dimensions", py::vectorize(&get_dimensions));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
