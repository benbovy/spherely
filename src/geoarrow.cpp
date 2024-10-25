#include <s2geography.h>

#include "arrow_abi.h"
#include "constants.hpp"
#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

py::array_t<PyObjectGeography> from_geoarrow(py::object input,
                                             bool oriented,
                                             bool planar,
                                             float tessellate_tolerance,
                                             py::object geometry_encoding) {
    py::tuple capsules = input.attr("__arrow_c_array__")();
    py::capsule schema_capsule = capsules[0];
    py::capsule array_capsule = capsules[1];

    const ArrowSchema* schema = static_cast<const ArrowSchema*>(schema_capsule);
    const ArrowArray* array = static_cast<const ArrowArray*>(array_capsule);

    s2geog::geoarrow::Reader reader;
    std::vector<std::unique_ptr<s2geog::Geography>> s2geog_vec;

    s2geog::geoarrow::ImportOptions options;
    options.set_oriented(oriented);
    if (planar) {
        auto tol = S1Angle::Radians(tessellate_tolerance / EARTH_RADIUS_METERS);
        options.set_tessellate_tolerance(tol);
    }
    if (geometry_encoding.is(py::none())) {
        reader.Init(schema, options);
    } else if (geometry_encoding.equal(py::str("WKT"))) {
        reader.Init(s2geog::geoarrow::Reader::InputType::kWKT, options);
    } else if (geometry_encoding.equal(py::str("WKB"))) {
        reader.Init(s2geog::geoarrow::Reader::InputType::kWKB, options);
    } else {
        throw std::invalid_argument("'geometry_encoding' should be one of None, 'WKT' or 'WKB'");
    }
    reader.ReadGeography(array, 0, array->length, &s2geog_vec);

    // Convert resulting vector to array of python objects
    auto result = py::array_t<PyObjectGeography>(array->length);
    py::buffer_info rbuf = result.request();
    py::object* rptr = static_cast<py::object*>(rbuf.ptr);

    py::ssize_t i = 0;
    for (auto& s2geog_ptr : s2geog_vec) {
        auto geog_ptr = std::make_unique<spherely::Geography>(std::move(s2geog_ptr));
        // return PyObjectGeography::from_geog(std::move(geog_ptr));
        rptr[i] = py::cast(std::move(geog_ptr));
        i++;
    }
    return result;
}

void init_geoarrow(py::module& m) {
    m.def("from_geoarrow",
          &from_geoarrow,
          py::arg("input"),
          py::pos_only(),
          py::kw_only(),
          py::arg("oriented") = false,
          py::arg("planar") = false,
          py::arg("tessellate_tolerance") = 100.0,
          py::arg("geometry_encoding") = py::none(),
          R"pbdoc(
        Create an array of geographies from an Arrow array object with a GeoArrow
        extension type.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        This functions accepts any Arrow array object implementing
        the `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_array__``
        method).

        .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

        Parameters
        ----------
        input : pyarrow.Array, Arrow array
            Any array object implementing the Arrow PyCapsule Protocol
            (i.e. has a ``__arrow_c_array__`` method). The type of the array
            should be one of the geoarrow geometry types.
        oriented : bool, default False
            Set to True if polygon ring directions are known to be correct
            (i.e., exterior rings are defined counter clockwise and interior
            rings are defined clockwise).
            By default (False), it will return the polygon with the smaller
            area.
        planar : bool, default False
            If set to True, the edges of linestrings and polygons are assumed
            to be linear on the plane. In that case, additional points will
            be added to the line while creating the geography objects, to
            ensure every point is within 100m of the original line.
            By default (False), it is assumed that the edges are spherical
            (i.e. represent the shortest path on the sphere between two points).
        tessellate_tolerance : float, default 100.0
            The maximum distance in meters that a point must be moved to
            satisfy the planar edge constraint. This is only used if `planar`
            is set to True.
        geometry_encoding : str, default None
            By default, the encoding is inferred from the GeoArrow extension
            type of the input array.
            However, for parsing WKT and WKB it is also possible to pass an
            Arrow array without geoarrow type but with a plain string or
            binary type, if specifying this keyword with "WKT" or "WKB",
            respectively.
    )pbdoc");
}
