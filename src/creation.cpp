#include "creation.hpp"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <s2/s2latlng.h>
#include <s2/s2loop.h>
#include <s2/s2point.h>
#include <s2/s2polygon.h>
#include <s2geography.h>
#include <s2geography/geography.h>

#include <array>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

//
// ---- S2geometry object creation functions.
//

S2Point make_s2point(double lng, double lat) {
    return S2LatLng::FromDegrees(lat, lng).ToPoint();
}

S2Point make_s2point(const std::pair<double, double> &vertex) {
    return S2LatLng::FromDegrees(vertex.first, vertex.second).ToPoint();
}

template <class V>
std::vector<S2Point> make_s2points(const std::vector<V> &vertices) {
    std::vector<S2Point> points(vertices.size());

    auto func = [](const V &vertex) {
        return make_s2point(vertex);
    };

    std::transform(vertices.begin(), vertices.end(), points.begin(), func);

    return std::move(points);
}

// create a S2Loop from coordinates or Point objects.
//
// Normalization (to CCW order for identifying the loop interior) and validation
// are both enabled by default.
//
// Additional normalization is made here:
// - if the input loop is already closed, remove one of the end nodes
//
// TODO: add option to skip normalization.
//
template <class V>
std::unique_ptr<S2Loop> make_s2loop(const std::vector<V> &vertices, bool check = true) {
    auto s2points = to_s2points(vertices);

    if (s2points.front() == s2points.back()) {
        s2points.pop_back();
    }

    auto loop_ptr = std::make_unique<S2Loop>();

    loop_ptr->set_s2debug_override(S2Debug::DISABLE);
    loop_ptr->Init(s2points);
    if (check && !loop_ptr->IsValid()) {
        std::stringstream err;
        S2Error s2err;
        err << "ring is not valid: ";
        loop_ptr->FindValidationError(&s2err);
        err << s2err.text();
        throw py::value_error(err.str());
    }

    loop_ptr->Normalize();

    return std::move(loop_ptr);
}

//
// ---- S2geometry / S2Geography / Spherely object wrapper utility functions
//

//
// ---- Spherely C++ Geography creation functions
//

// std::unique_ptr<Geography> create_point2(double lng, double lat) {
//     return make_geography<s2geog::PointGeography>(to_s2point2(std::make_pair(lat, lng)));
// }

// template <class V>
// std::unique_ptr<Geography> create_multipoint2(const std::vector<V> &pts) {
//     return make_geography2<s2geog::PointGeography>(to_s2points(pts));
// }

// template <class V>
// std::unique_ptr<Geography> create_linestring2(const std::vector<V> &pts) {
//     auto s2points = to_s2points(pts);
//     auto polyline_ptr = std::make_unique<S2Polyline>(s2points);

//     return make_geography2<s2geog::PolylineGeography>(std::move(polyline_ptr));
// }

// std::unique_ptr<Geography> create_multilinestring2(const std::vector<LineString *> &lines) {
//     std::vector<std::unique_ptr<S2Polyline>> polylines(lines.size());

//     auto func = [](const LineString *line_ptr) {
//         S2Polyline *cloned_ptr(line_ptr->s2polyline().Clone());
//         return std::make_unique<S2Polyline>(std::move(*cloned_ptr));
//     };

//     std::transform(lines.begin(), lines.end(), polylines.begin(), func);

//     return make_geography2<s2geog::PolylineGeography>(std::move(polylines));
// }

// template <class V>
// std::unique_ptr<Geography> create_multilinestring2(const std::vector<std::vector<V>> &lines) {
//     std::vector<std::unique_ptr<S2Polyline>> polylines(lines.size());

//     auto func = [](const std::vector<V> &pts) {
//         auto s2points = to_s2points(pts);
//         return std::make_unique<S2Polyline>(s2points);
//     };

//     std::transform(lines.begin(), lines.end(), polylines.begin(), func);

//     return make_geography2<s2geog::PolylineGeography>(std::move(polylines));
// }

// template <class V>
// std::unique_ptr<Geography> create_polygon2(
//     const std::vector<V> &shell,
//     const std::optional<std::vector<std::vector<V>>> &holes) {
//     std::vector<std::unique_ptr<S2Loop>> loops;
//     loops.push_back(make_s2loop(shell, false));

//     if (holes.has_value()) {
//         for (const auto &ring : holes.value()) {
//             loops.push_back(make_s2loop(ring, false));
//         }
//     }

//     auto polygon_ptr = std::make_unique<S2Polygon>();
//     polygon_ptr->set_s2debug_override(S2Debug::DISABLE);
//     polygon_ptr->InitNested(std::move(loops));

//     // Note: this also checks each loop of the polygon
//     if (!polygon_ptr->IsValid()) {
//         std::stringstream err;
//         S2Error s2err;
//         err << "polygon is not valid: ";
//         polygon_ptr->FindValidationError(&s2err);
//         err << s2err.text();
//         throw py::value_error(err.str());
//     }

//     return make_geography2<s2geog::PolygonGeography>(std::move(polygon_ptr));
// }

// std::unique_ptr<GeographyCollection> create_collection2(const std::vector<Geography *> &features)
// {
//     std::vector<std::unique_ptr<s2geog::Geography>> features_copy;
//     features_copy.reserve(features.size());

//     for (const auto &feature_ptr : features) {
//         features_copy.push_back(clone_s2geography(feature_ptr->geog()));
//     }

//     return std::make_unique<GeographyCollection>(
//         std::make_unique<s2geog::GeographyCollection>(std::move(features_copy)));
//     ;
// }

//
// ---- Spherely Python Geography creation functions
//

// support array_like (list of lists) of longitude, latitude coordinates
using VectorLngLat = std::vector<std::pair<double, double>>;
PYBIND11_MAKE_OPAQUE(VectorLngLat)

PyObjectGeography point(double longitude, double latitude) {
    return make_py_geography<s2geog::PointGeography>(make_s2point(longitude, latitude));
}

py::array_t<PyObjectGeography> points(const VectorLngLat &coords) {
    auto npoints = static_cast<py::ssize_t>(coords.size());
    auto points = py::array_t<PyObjectGeography>(npoints);

    py::buffer_info buf = points.request();
    py::object *data = static_cast<py::object *>(buf.ptr);

    for (size_t i = 0; i < coords.size(); i++) {
        auto lnglat = coords[i];
        auto point_ptr = point(lnglat.first, lnglat.second);
        data[i] = py::cast(std::move(point_ptr));
    }

    return points;
}

py::array_t<PyObjectGeography> points(const py::array_t<double> &coords) {
    auto coords_data = coords.unchecked<2>();

    if (coords_data.shape(1) != 2) {
        throw std::runtime_error("coords array must be of shape (N, 2)");
    }

    auto npoints = coords_data.shape(0);
    auto points = py::array_t<PyObjectGeography>(npoints);

    py::buffer_info buf = points.request();
    py::object *data = static_cast<py::object *>(buf.ptr);

    for (py::ssize_t i = 0; i < npoints; i++) {
        auto point_ptr = point(coords_data(i, 0), coords_data(i, 1));
        data[i] = py::cast(std::move(point_ptr));
    }

    return points;
}

//
// ---- Geography creation Python bindings
//

void init_creation(py::module &m) {
    py::bind_vector<VectorLngLat>(m, "_VectorLngLat");
    py::implicitly_convertible<py::list, VectorLngLat>();

    m.def("points",
          py::vectorize(&point),
          py::arg("longitude"),
          py::arg("latitude"),
          R"pbdoc(
        Create an array of points.

        Parameters
        ----------
        longitude : array_like
            longitude coordinate(s), in degrees.
        latitude : array_like
            latitude coordinate(s), in degrees.

    )pbdoc");

    m.def("points",
          py::overload_cast<const VectorLngLat &>(&points),
          py::arg("coords"),
          R"pbdoc(
        Create an array of points.

        Parameters
        ----------
        coords : array_like
            A array of longitude, latitude coordinate tuples (i.e., with shape (N, 2)).

    )pbdoc");

    m.def("points",
          py::overload_cast<const py::array_t<double> &>(&points),
          py::arg("coords"),
          R"pbdoc(
        Create an array of points.

        Parameters
        ----------
        coords : array_like
            A array of longitude, latitude coordinate tuples (i.e., with shape (N, 2)).

    )pbdoc");
}
