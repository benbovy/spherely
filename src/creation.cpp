#include "creation.hpp"

#include <pybind11/attr.h>
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
#include <functional>
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
        // We raise an exception that is caught when trying to build a Geography from an empty point
        // TODO: what do we want here if this is reused in other contexts?
        // Return S2Point::NaN() or S2Point()?
        throw EmptyGeographyException("cannot create s2geometry point from empty POINT Geography");
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
    auto s2points = make_s2points(vertices);

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
        data[i] = std::move(point_ptr);
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
    if (pts.size() == 1) {
        throw py::value_error("linestring is not valid: it must have at least 2 vertices");
    }

    try {
        auto s2points = make_s2points(pts);
        auto polyline_ptr = std::make_unique<S2Polyline>(s2points);
        return make_geography<s2geog::PolylineGeography>(std::move(polyline_ptr));
    } catch (const EmptyGeographyException &error) {
        throw py::value_error("can't create LineString with empty component");
    }
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

template <class V>
std::unique_ptr<Geography> polygon(const std::vector<V> &shell,
                                   const std::optional<std::vector<std::vector<V>>> &holes) {
    // fastpath empty polygon
    if (shell.empty()) {
        return make_geography<s2geog::PolygonGeography>(std::make_unique<S2Polygon>());
    }

    std::vector<std::unique_ptr<S2Loop>> loops;

    try {
        loops.push_back(make_s2loop(shell, false));
    } catch (const EmptyGeographyException &error) {
        throw py::value_error("can't create Polygon with empty component");
    }

    if (holes.has_value()) {
        for (const auto &ring : holes.value()) {
            loops.push_back(make_s2loop(ring, false));
        }
    }

    auto polygon_ptr = std::make_unique<S2Polygon>();
    polygon_ptr->set_s2debug_override(S2Debug::DISABLE);
    polygon_ptr->InitNested(std::move(loops));

    // Note: this also checks each loop of the polygon
    if (!polygon_ptr->IsValid()) {
        std::stringstream err;
        S2Error s2err;
        err << "polygon is not valid: ";
        polygon_ptr->FindValidationError(&s2err);
        err << s2err.text();
        throw py::value_error(err.str());
    }

    return make_geography<s2geog::PolygonGeography>(std::move(polygon_ptr));
}

std::unique_ptr<Geography> geography_collection(const std::vector<Geography *> &features) {
    std::vector<std::unique_ptr<s2geog::Geography>> features_copy;
    features_copy.reserve(features.size());

    for (const auto &feature_ptr : features) {
        features_copy.push_back(feature_ptr->clone_geog());
    }

    return make_geography<s2geog::GeographyCollection>(std::move(features_copy));
}

//
// ---- Geography creation Python bindings
//

void init_creation(py::module &m) {
    py::options options;
    options.disable_function_signatures();

    // ----- scalar Geography creation functions

    m.def(
        "point",
        [](py::object longitude, py::object latitude) {
            if (longitude.is_none() && latitude.is_none()) {
                // empty point
                return make_geography(std::make_unique<s2geog::PointGeography>());
            } else if (longitude.is_none() || latitude.is_none()) {
                throw py::type_error(
                    "can only provide None (empty point) or float values for both longitude and "
                    "latitude");
            } else {
                return make_geography<s2geog::PointGeography>(
                    make_s2point(longitude.cast<double>(), latitude.cast<double>()));
            }
        },
        py::arg("longitude") = py::none(),
        py::arg("latitude") = py::none(),
        R"pbdoc(point(longitude: float | None = None, latitude: float | None = None) -> Geography
        Create a POINT geography.

        Parameters
        ----------
        longitude : float, optional
            longitude coordinate, in degrees.
        latitude : float, optional
            latitude coordinate, in degrees.

    )pbdoc");

    m.def("multipoint",
          &multipoint<std::pair<double, double>>,
          py::arg("points"),
          R"pbdoc(multipoint(points: Sequence) -> Geography
        Create a MULTIPOINT geography.

        Parameters
        ----------
        points : sequence
            A sequence of (longitude, latitude) coordinates (in degrees) or
            POINT :class:`~spherely.Geography` objects.

    )pbdoc")
        .def("multipoint", &multipoint<Geography *>, py::arg("points"));

    m.def(
         "linestring",
         [](py::none) { return make_geography(std::make_unique<s2geog::PolylineGeography>()); },
         py::arg("vertices") = py::none(),
         R"pbdoc(linestring(vertices: Sequence | None = None) -> Geography
        Create a LINESTRING geography.

        Parameters
        ----------
        vertices : sequence, optional
            A sequence of (longitude, latitude) coordinates (in degrees) or
            POINT :class:`~spherely.Geography` objects.

        )pbdoc")
        .def("linestring", &linestring<std::pair<double, double>>, py::arg("vertices"))
        .def("linestring", &linestring<Geography *>, py::arg("vertices"));

    m.def("multilinestring",
          &multilinestring<std::pair<double, double>>,
          py::arg("lines"),
          R"pbdoc(multilinestring(lines: Sequence) -> Geography
        Create a MULTILINESTRING geography.

        Parameters
        ----------
        lines : sequence
            A sequence of sequences of (longitude, latitude) coordinates (in degrees) or
            a sequence of sequences of POINT :class:`~spherely.Geography` objects or
            a sequence of LINESTRING :class:`~spherely.Geography` objects.

    )pbdoc")
        .def("multilinestring", &multilinestring<Geography *>, py::arg("lines"))
        .def(
            "multilinestring",
            [](const std::vector<Geography *> lines) { return multilinestring(lines); },
            py::arg("lines"));

    m.def(
         "polygon",
         [](py::none, py::none) {
             // TODO: remove explicit creation of S2Polygon, see
             // https://github.com/paleolimbot/s2geography/pull/31
             auto empty_poly = std::make_unique<S2Polygon>();
             return make_geography(
                 std::make_unique<s2geog::PolygonGeography>(std::move(empty_poly)));
         },
         py::arg("shell") = py::none(),
         py::arg("holes") = py::none(),
         R"pbdoc(polygon(shell: Sequence | None = None, holes: Sequence | None = None) -> Geography
        Create a POLYGON geography.

        Parameters
        ----------
        shell : sequence, optional
            A sequence of (longitude, latitude) coordinates (in degrees) or
            POINT :class:`~spherely.Geography` objects  representing the vertices of the polygon.
        holes : sequence, optional
            A list of sequences of objects where each sequence satisfies the same
            requirements as the ``shell`` argument.

    )pbdoc")
        .def("polygon",
             &polygon<std::pair<double, double>>,
             py::arg("shell"),
             py::arg("holes") = py::none())
        .def("polygon", &polygon<Geography *>, py::arg("shell"), py::arg("holes") = py::none());

    m.def("geography_collection",
          &geography_collection,
          py::arg("geographies"),
          R"pbdoc(geography_collection(geographies: Sequence) -> Geography
        Create a GEOGRAPHYCOLLECTION geography from arbitrary geographies.

        Parameters
        ----------
        geographies : sequence
            A sequence of :class:`~spherely.Geography` objects.

    )pbdoc");

    // ----- vectorized Geography creation functions

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
}
