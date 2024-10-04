#include "geography.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <s2/s2latlng.h>
#include <s2/s2loop.h>
#include <s2/s2point.h>
#include <s2/s2polygon.h>
#include <s2geography.h>

#include <memory>
#include <optional>
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
    pygeography_types.value("POLYGON", GeographyType::Polygon);
    pygeography_types.value("MULTIPOLYGON", GeographyType::MultiPolygon);
    pygeography_types.value("MULTIPOINT", GeographyType::MultiPoint);
    pygeography_types.value("MULTILINESTRING", GeographyType::MultiLineString);
    pygeography_types.value("GEOGRAPHYCOLLECTION", GeographyType::GeographyCollection);

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

    // Temp test

    // m.def("nshape", &num_shapes);
    // m.def("create", &create);

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
