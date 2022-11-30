#include "geography.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <s2/s2latlng.h>
#include <s2geography.h>

#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace s2shapely;

/*
** Geography factories
*/

class PointFactory {
public:
    static std::unique_ptr<Point> FromLatLonDegrees(double lat_degrees,
                                                    double lon_degrees) {
        auto latlng = S2LatLng::FromDegrees(lat_degrees, lon_degrees);
        S2GeographyPtr s2geog_ptr =
            std::make_unique<s2geog::PointGeography>(S2Point(latlng));
        std::unique_ptr<Point> point_ptr =
            std::make_unique<Point>(std::move(s2geog_ptr));

        return point_ptr;
    }
};

class LineStringFactory {
public:
    using LatLonCoords = std::vector<std::pair<double, double>>;

    static std::unique_ptr<LineString> FromLatLonCoords(LatLonCoords coords) {
        std::vector<S2LatLng> latlng_pts;
        for (auto &latlng : coords) {
            latlng_pts.push_back(
                S2LatLng::FromDegrees(latlng.first, latlng.second));
        }

        auto polyline = std::make_unique<S2Polyline>(latlng_pts);
        S2GeographyPtr s2geog_ptr =
            std::make_unique<s2geog::PolylineGeography>(std::move(polyline));

        return std::make_unique<LineString>(std::move(s2geog_ptr));
    }
};

/*
** Temporary testing Numpy-vectorized API (TODO: remove)
*/

py::array_t<int> get_dimensions_direct(const py::array_t<PyObjectGeography> geographies) {
    py::buffer_info buf = geographies.request();

    auto result = py::array_t<int>(buf.size);
    py::buffer_info result_buf = result.request();
    int *rptr = static_cast<int *>(result_buf.ptr);

    py::gil_scoped_release();
    for (size_t i = 0; i < buf.size; i++) {
        auto geog_ptr = (*geographies.data(i)).as_geog_ptr();
        rptr[i] = geog_ptr->dimension();
    }

    return result;
}

py::array_t<PyObjectGeography> create(py::array_t<double> xs,
                                      py::array_t<double> ys) {
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
        //rptr[i] = PyObjectGeography::as_py_object(std::move(point_ptr));
        rptr[i] = py::cast(std::move(point_ptr));
    }

    return result;
}

py::array_t<double> array_add(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info abuf = a.request(), bbuf = b.request();

    auto result = py::array_t<double>(abuf.size);
    py::buffer_info rbuf = result.request();

    double *aptr = static_cast<double *>(abuf.ptr);
    double *bptr = static_cast<double *>(bbuf.ptr);
    double *rptr = static_cast<double *>(rbuf.ptr);

    py::gil_scoped_release();
    for (size_t i = 0; i < abuf.size; i++) {
        rptr[i] = aptr[i] + bptr[i];
    }

    return result;
}

void get_dimensions_pure_cpp(py::array_t<double> xs, py::array_t<double> ys) {
    py::buffer_info xbuf = xs.request(), ybuf = ys.request();
    if (xbuf.ndim != 1 || ybuf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be one");
    }
    if (xbuf.size != ybuf.size) {
        throw std::runtime_error("Input shapes must match");
    }

    double *xptr = static_cast<double *>(xbuf.ptr);
    double *yptr = static_cast<double *>(ybuf.ptr);

    std::vector<std::unique_ptr<Point>> input;

    for (size_t i = 0; i < xbuf.shape[0]; i++) {
        input.push_back(PointFactory::FromLatLonDegrees(xptr[i], yptr[i]));
    }

    std::vector<int> result;

    for (size_t i = 0; i < xbuf.shape[0]; i++) {
        result.push_back(input[i]->dimension());
    }
}


py::array_t<PyObjectGeography> create_no_cast(py::array_t<double> xs, py::array_t<double> ys) {
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
        // skip this cast
        //rptr[i] = PyObjectGeography::as_py_object(std::move(point_ptr));
        //rptr[i] = reinterpret_cast<py::object>(std::move(point_ptr));
        rptr[i] = py::object();
    }

    return result;
}

int dummy(PyObjectGeography obj) {
    obj.as_geog_ptr();
    return 0;
}
int dummy_no_cast(PyObjectGeography obj) { return 0; }

/*
** Geography properties
*/

std::int8_t get_type_id(PyObjectGeography obj) {
    return static_cast<std::int8_t>(obj.as_geog_ptr()->geog_type());
}

int get_dimensions(PyObjectGeography obj) {
    return obj.as_geog_ptr()->dimension();
}

/*
** Geography utils
*/

bool is_geography(PyObjectGeography obj) { return obj.is_geog_ptr(); }

bool is_prepared(PyObjectGeography obj) {
    return obj.as_geog_ptr()->has_index();
}

PyObjectGeography prepare(PyObjectGeography obj) {
    // triggers index creation if not yet built
    obj.as_geog_ptr()->geog_index();
    return obj;
}

PyObjectGeography destroy_prepared(PyObjectGeography obj) {
    obj.as_geog_ptr()->reset_index();
    return obj;
}


double add(double x, double y) {
    return x + y;
}

void init_geography(py::module &m) {
    // Geography types

    py::enum_<GeographyType>(m, "GeographyType")
        .value("NONE", GeographyType::None)
        .value("POINT", GeographyType::Point)
        .value("LINESTRING", GeographyType::LineString);

    // Geography classes

    py::class_<Geography>(m, "Geography")
        .def_property_readonly("dimensions", &Geography::dimension)
        .def_property_readonly("nshape", &Geography::num_shapes)
        .def("__repr__", [](const Geography &geog) {
            s2geog::WKTWriter writer;
            return writer.write_feature(geog.geog());
        });

    py::class_<Point, Geography>(m, "Point")
        .def(py::init(&PointFactory::FromLatLonDegrees));

    py::class_<LineString, Geography>(m, "LineString")
        .def(py::init(&LineStringFactory::FromLatLonCoords));

    // Temp test

    m.def("get_dimensions_direct", &get_dimensions_direct);
    m.def("get_dimensions_pure_cpp", &get_dimensions_pure_cpp);
    m.def("create", &create);
    m.def("create_no_cast", &create_no_cast);
    m.def("dummy", py::vectorize(&dummy));
    m.def("dummy_no_cast", py::vectorize(&dummy_no_cast));
    m.def("add", py::vectorize(&add));
    m.def("add", py::vectorize(&add), py::call_guard<py::gil_scoped_release>());
    m.def("array_add", &array_add);

    // Geography properties

    m.def("get_type_id", py::vectorize(&get_type_id));
    m.def("get_dimensions", py::vectorize(&get_dimensions));

    // Geography utils

    m.def("is_geography", py::vectorize(&is_geography));
    m.def("is_prepared", py::vectorize(&is_prepared));
    m.def("prepare", py::vectorize(&prepare));
    m.def("destroy_prepared", py::vectorize(&destroy_prepared));
}
