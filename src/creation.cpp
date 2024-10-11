#include "creation.hpp"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <s2/s2latlng.h>
#include <s2/s2loop.h>
#include <s2/s2point.h>
#include <s2/s2polygon.h>
#include <s2/s2polyline.h>
#include <s2geography.h>
#include <s2geography/geography.h>

#include <array>
#include <memory>
#include <sstream>
#include <stdexcept>
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

S2Point make_s2point(const std::pair<double, double> &point) {
    return S2LatLng::FromDegrees(point.second, point.first).ToPoint();
}

S2Point make_s2point(const Geography *point_ptr) {
    check_geog_type(*point_ptr, GeographyType::Point);
    auto s2geog_obj = static_cast<const s2geog::PointGeography &>(point_ptr->geog());

    if (s2geog_obj.Points().empty()) {
        throw EmptyGeographyException("cannot create s2geometry point from empty POINT Geography");
        // TODO: what do we want here? NaN or S2Point default constructor? It depends?
        // return S2Point::NaN();
    }

    return s2geog_obj.Points()[0];
}

template <class V>
std::vector<S2Point> make_s2points(const std::vector<V> &points) {
    std::vector<S2Point> s2points(points.size());

    auto func = [](const V &vertex) {
        return make_s2point(vertex);
    };

    std::transform(points.begin(), points.end(), s2points.begin(), func);

    return std::move(s2points);
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

PyObjectGeography point(double longitude, double latitude) {
    return make_py_geography<s2geog::PointGeography>(make_s2point(longitude, latitude));
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

template <class V>
std::unique_ptr<Geography> multipoint(const std::vector<V> &pts) {
    try {
        return make_geography<s2geog::PointGeography>(make_s2points(pts));
    } catch (const EmptyGeographyException &error) {
        throw py::value_error("can't create MultiPoint with empty component");
    }
}

template <class V>
std::unique_ptr<Geography> linestring(const std::vector<V> &pts) {
    auto s2points = make_s2points(pts);
    auto polyline_ptr = std::make_unique<S2Polyline>(s2points);

    return make_geography<s2geog::PolylineGeography>(std::move(polyline_ptr));
}

template <class V>
std::unique_ptr<Geography> multilinestring(const std::vector<std::vector<V>> &lines) {
    std::vector<std::unique_ptr<S2Polyline>> polylines(lines.size());

    auto func = [](const std::vector<V> &pts) {
        try {
            auto s2points = make_s2points(pts);
            return std::make_unique<S2Polyline>(s2points);
        } catch (const EmptyGeographyException &error) {
            throw py::value_error("can't create MultiLineString with empty component");
        }
    };

    std::transform(lines.begin(), lines.end(), polylines.begin(), func);

    return make_geography<s2geog::PolylineGeography>(std::move(polylines));
}

std::unique_ptr<Geography> multilinestring(const std::vector<Geography *> &lines) {
    std::vector<std::unique_ptr<S2Polyline>> polylines(lines.size());

    auto func = [](const Geography *line_ptr) {
        check_geog_type(*line_ptr, GeographyType::LineString);

        auto s2geog_ptr = static_cast<const s2geog::PolylineGeography *>(&line_ptr->geog());
        auto polylines_ptr = &s2geog_ptr->Polylines();

        if (polylines_ptr->empty()) {
            throw py::value_error("can't create MultiLineString with empty component");
        }

        S2Polyline *cloned_ptr((*polylines_ptr)[0]->Clone());
        return std::make_unique<S2Polyline>(std::move(*cloned_ptr));
    };

    std::transform(lines.begin(), lines.end(), polylines.begin(), func);

    return make_geography<s2geog::PolylineGeography>(std::move(polylines));
}

//
// ---- Geography creation Python bindings
//

void init_creation(py::module &m) {
    m.def("point",
          &point,
          py::arg("longitude"),
          py::arg("latitude"),
          R"pbdoc(
        Create a single point.

        Parameters
        ----------
        longitude : float
            longitude coordinate, in degrees.
        latitude : float
            latitude coordinate, in degrees.

    )pbdoc");

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
          &points,
          py::arg("coords"),
          R"pbdoc(
        Create an array of points.

        Parameters
        ----------
        coords : array_like
            A array of longitude, latitude coordinate tuples (i.e., with shape (N, 2)).

    )pbdoc");

    m.def("multipoint",
          &multipoint<std::pair<double, double>>,
          py::arg("points"),
          R"pbdoc(
        Create a MULTIPOINT feature.

        Parameters
        ----------
        points : sequence
            A sequence of (longitude, latitude) coordinates, in degrees.

    )pbdoc");

    m.def("multipoint",
          &multipoint<Geography *>,
          py::arg("points"),
          R"pbdoc(
        Create a MULTIPOINT feature.

        Parameters
        ----------
        points : sequence
            A sequence of POINT :class:`~spherely.Geography` objects.

    )pbdoc");

    m.def("linestring",
          &linestring<std::pair<double, double>>,
          py::arg("vertices"),
          R"pbdoc(
        Create a linestring.

        Parameters
        ----------
        vertices : sequence
            A sequence of (longitude, latitude) coordinates, in degrees.

    )pbdoc");

    m.def("linestring",
          &linestring<Geography *>,
          py::arg("vertices"),
          R"pbdoc(
        Create a linestring.

        Parameters
        ----------
        vertices : sequence
            A sequence of POINT :class:`~spherely.Geography` objects.

    )pbdoc");

    m.def("multilinestring",
          &multilinestring<std::pair<double, double>>,
          py::arg("lines"),
          R"pbdoc(
        Create a MULTILINESTRING feature.

        Parameters
        ----------
        lines : sequence
            A sequence of sequences of (longitude, latitude) coordinates, in degrees.

    )pbdoc");

    m.def("multilinestring",
          &multilinestring<Geography *>,
          py::arg("lines"),
          R"pbdoc(
        Create a MULTILINESTRING feature.

        Parameters
        ----------
        lines : sequence
            A sequence of sequences of POINT :class:`~spherely.Geography` objects.

    )pbdoc");

    m.def(
        "multilinestring",
        [](const std::vector<Geography *> lines) { return multilinestring(lines); },
        py::arg("lines"),
        R"pbdoc(
        Create a MULTILINESTRING feature.

        Parameters
        ----------
        lines : sequence
            A sequence of LINESTRING :class:`~spherely.Geography` objects.

    )pbdoc");
}