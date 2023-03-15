#include "geography.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <s2/s2latlng.h>
#include <s2/s2loop.h>
#include <s2/s2point.h>
#include <s2/s2polygon.h>
#include <s2geography.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "pybind11.hpp"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

py::detail::type_info *PyObjectGeography::geography_tinfo = nullptr;

/*
** Geography factories
*/

// Used in Geography constructors to get a point either from a tuple of
// coordinates or an existing Point object.
S2Point to_s2point(const std::pair<double, double> &vertex) {
    return S2LatLng::FromDegrees(vertex.first, vertex.second).ToPoint();
}

S2Point to_s2point(const Point *vertices) {
    return vertices->s2point();
}

// create a S2Loop from coordinates or Point objects.
//
// Normalization (to CCW order for identifying the loop interior) and validation
// are both enabled by default.
//
// TODO: add options to skip normalization and/or validation.
//
template <class V>
std::unique_ptr<S2Loop> make_s2loop(const std::vector<V> &vertices) {
    std::vector<S2Point> points(vertices.size());

    std::transform(vertices.begin(), vertices.end(), points.begin(), [](const V &vertex) {
        return to_s2point(vertex);
    });

    auto loop_ptr = std::make_unique<S2Loop>();

    loop_ptr->set_s2debug_override(S2Debug::DISABLE);
    loop_ptr->Init(points);
    if (!loop_ptr->IsValid()) {
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

/*
** Helper to create Geography object wrappers.
**
** @tparam T The Geography type (spherely wrapper)
** @tparam S The S2Geometry type
*/
template <class T, class S, std::enable_if_t<std::is_base_of<Geography, T>::value, bool> = true>
std::unique_ptr<T> make_geography(S &&s2_obj) {
    using S2GeographyType = typename T::S2GeographyType;

    S2GeographyPtr s2geog_ptr = std::make_unique<S2GeographyType>(std::forward<S>(s2_obj));
    return std::make_unique<T>(std::move(s2geog_ptr));
}

class PointFactory {
public:
    static std::unique_ptr<Point> FromLatLonDegrees(double lat_degrees, double lon_degrees) {
        auto latlng = S2LatLng::FromDegrees(lat_degrees, lon_degrees);

        return make_geography<Point>(S2Point(latlng));
    }

    // TODO: from LatLonRadians
};

template <class V>
static std::unique_ptr<LineString> create_linestring(const std::vector<V> &coords) {
    std::vector<S2Point> pts(coords.size());

    std::transform(coords.begin(), coords.end(), pts.begin(), [](const V &vertex) {
        return to_s2point(vertex);
    });

    auto polyline_ptr = std::make_unique<S2Polyline>(pts);

    return make_geography<LineString>(std::move(polyline_ptr));
}

template <class V>
static std::unique_ptr<LinearRing> create_linearring(const std::vector<V> &coords) {
    return make_geography<LinearRing>(*make_s2loop(coords));
}

template <class V>
static std::unique_ptr<spherely::Polygon> create_polygon(const std::vector<V> &shell) {
    std::vector<std::unique_ptr<S2Loop>> loops;
    loops.push_back(make_s2loop(shell));

    auto polygon_ptr = std::make_unique<S2Polygon>();
    // TODO: maybe add an option to skip validity checks
    polygon_ptr->set_s2debug_override(S2Debug::DISABLE);
    polygon_ptr->InitOriented(std::move(loops));

    return make_geography<spherely::Polygon>(std::move(polygon_ptr));
}

/*
** Temporary testing Numpy-vectorized API (TODO: remove)
*/

py::array_t<int> num_shapes(const py::array_t<PyObjectGeography> geographies) {
    py::buffer_info buf = geographies.request();

    auto result = py::array_t<int>(buf.size);
    py::buffer_info result_buf = result.request();
    int *rptr = static_cast<int *>(result_buf.ptr);

    for (py::ssize_t i = 0; i < buf.size; i++) {
        auto geog_ptr = (*geographies.data(i)).as_geog_ptr();
        rptr[i] = geog_ptr->num_shapes();
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

    for (py::ssize_t i = 0; i < xbuf.shape[0]; i++) {
        auto point_ptr = PointFactory::FromLatLonDegrees(xptr[i], yptr[i]);
        // rptr[i] = PyObjectGeography::as_py_object(std::move(point_ptr));
        rptr[i] = py::cast(std::move(point_ptr));
    }

    return result;
}

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

bool is_geography(PyObjectGeography obj) {
    return obj.is_geog_ptr();
}

/*
** Geography creation
*/

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

void init_geography(py::module &m) {
    // Geography types

    auto pygeography_types = py::enum_<GeographyType>(m, "GeographyType", R"pbdoc(
        The enumeration of Geography types
    )pbdoc");

    pygeography_types.value("NONE", GeographyType::None);
    pygeography_types.value("POINT", GeographyType::Point);
    pygeography_types.value("LINESTRING", GeographyType::LineString);
    pygeography_types.value("LINEARRING", GeographyType::LinearRing);
    pygeography_types.value("POLYGON", GeographyType::Polygon);

    // Geography classes

    auto pygeography = py::class_<Geography>(m, "Geography", R"pbdoc(
        Base class for all geography types.

        Cannot be instanciated directly.

    )pbdoc");

    pygeography.def_property_readonly("dimensions",
                                      &Geography::dimension,
                                      R"pbdoc(
        Returns the inherent dimensionality of a geometry.

        The inherent dimension is 0 for points, 1 for linestrings and 2 for
        polygons. For geometrycollections it is the max of the containing elements.
        Empty collections and None values return -1.

    )pbdoc");

    pygeography.def_property_readonly("nshape", &Geography::num_shapes, R"pbdoc(
        Returns the number of elements in the collection, or 1 for simple geography
        objects.

    )pbdoc");

    pygeography.def("__repr__", [](const Geography &geog) {
        s2geog::WKTWriter writer;
        return writer.write_feature(geog.geog());
    });

    auto pypoint = py::class_<Point, Geography>(m, "Point", R"pbdoc(
        A geography type that represents a single coordinate with lat,lon or x,y,z values.

        A point is a zero-dimensional feature and has zero length and zero area.

        Parameters
        ----------
        lat : float
            latitude coordinate, in degrees
        lon : float
            longitude coordinate, in degrees

    )pbdoc");

    pypoint.def(py::init(&PointFactory::FromLatLonDegrees), py::arg("lat"), py::arg("lon"));

    auto pylinestring = py::class_<LineString, Geography>(m, "LineString", R"pbdoc(
        A geography type composed of one or more arc (geodesic) segments.

        A LineString is a one-dimensional feature and has a non-zero length but
        zero area. A LineString is not closed.

        Parameters
        ----------
        coordinates : list
            A sequence of (lat, lon) tuple coordinates or :py:class:`Point` objects
            for each vertex.

    )pbdoc");

    pylinestring.def(py::init(&create_linestring<std::pair<double, double>>),
                     py::arg("coordinates"));
    pylinestring.def(py::init(&create_linestring<Point *>), py::arg("coordinates"));

    auto pylinearring = py::class_<LinearRing, Geography>(m, "LinearRing", R"pbdoc(
        A geography type composed of two or more arc (geodesic) segments
        that form a closed loop.

        A LinearRing is a closed, one-dimensional feature. It must have at least 3
        vertices, cannot crosses itself and cannot have duplicate vertices.
        Arcs of length 180 degrees are not allowed.

        The ring is automatically closed. There is no need to specify a final coordinate
        pair or point identical to the first.

        Parameters
        ----------
        coordinates : list
            A sequence of (lat, lon) tuple coordinates or :py:class:`Point` objects
            for each vertex.

    )pbdoc");

    pylinearring.def(py::init(&create_linearring<std::pair<double, double>>),
                     py::arg("coordinates"));
    pylinearring.def(py::init(&create_linearring<Point *>), py::arg("coordinates"));

    pylinearring.def("__repr__", [](const LinearRing &geog) {
        return static_cast<const s2geog::ClosedPolylineGeography *>(&geog.geog())->wkt(6);
    });

    auto pypolygon = py::class_<spherely::Polygon, Geography>(m, "Polygon", R"pbdoc(
        A geography type representing an area that is enclosed by a linear ring.

        A polygon is a two-dimensional feature and has a non-zero area.

        Parameters
        ----------
        shell : list
            A sequence of (lat, lon) tuple coordinates or :py:class:`Point` objects
            for each vertex of the polygon.

    )pbdoc");

    pypolygon.def(py::init(&create_polygon<std::pair<double, double>>), py::arg("shell"));
    pypolygon.def(py::init(&create_polygon<Point *>), py::arg("shell"));

    // Temp test

    m.def("nshape", &num_shapes);
    m.def("create", &create);

    // Geography properties

    m.def("get_type_id",
          py::vectorize(&get_type_id),
          py::arg("geography"),
          R"pbdoc(
        Returns the type ID of a geography.

        - None (missing) is -1
        - POINT is 0
        - LINESTRING is 1

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).

    )pbdoc");

    m.def("get_dimensions", py::vectorize(&get_dimensions), py::arg("geography"), R"pbdoc(
        Returns the inherent dimensionality of a geography.

        The inherent dimension is 0 for points, 1 for linestrings and 2 for
        polygons. For geometrycollections it is the max of the containing elements.
        Empty collections and None values return -1.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");

    // Geography utils

    m.def("is_geography",
          py::vectorize(&is_geography),
          py::arg("obj"),
          R"pbdoc(
        Returns True if the object is a :py:class:`Geography`, False otherwise.

        Parameters
        ----------
        obj : any or array_like
            Any object.

    )pbdoc");

    // Geography creation

    m.def("is_prepared",
          py::vectorize(&is_prepared),
          py::arg("geography"),
          R"pbdoc(
        Returns True if the geography object is "prepared", False otherwise.

        A prepared geography is a normal geography with added information such as
        an index on the line segments. This improves the performance of many operations.

        Note that it is not necessary to check if a geography is already prepared
        before preparing it. It is more efficient to call prepare directly
        because it will skip geographies that are already prepared.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s)

        See Also
        --------
        prepare
        destroy_prepared

    )pbdoc");

    m.def("prepare",
          py::vectorize(&prepare),
          py::arg("geography"),
          R"pbdoc(
        Prepare a geography, improving performance of other operations.

        A prepared geography is a normal geography with added information such as
        an index on the line segments. This improves the performance of the
        following operations.

        This function does not recompute previously prepared geographies; it is
        efficient to call this function on an array that partially contains
        prepared geographies.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s)

        See Also
        --------
        is_prepared
        destroy_prepared

    )pbdoc");

    m.def("destroy_prepared",
          py::vectorize(&destroy_prepared),
          py::arg("geography"),
          R"pbdoc(
        Destroy the prepared part of a geography, freeing up memory.

        Note that the prepared geography will always be cleaned up if the
        geography itself is dereferenced. This function needs only be called in
        very specific circumstances, such as freeing up memory without losing
        the geographies, or benchmarking.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s)

        See Also
        --------
        is_prepared
        prepare

    )pbdoc");
}
