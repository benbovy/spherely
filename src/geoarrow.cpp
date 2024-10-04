#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

#ifdef __cplusplus
extern "C" {
#endif

// Extra guard for versions of Arrow without the canonical guard
#ifndef ARROW_FLAG_DICTIONARY_ORDERED

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
    // Array type description
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;

    // Release callback
    void (*release)(struct ArrowSchema*);
    // Opaque producer-specific data
    void* private_data;
};

struct ArrowArray {
    // Array data description
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;

    // Release callback
    void (*release)(struct ArrowArray*);
    // Opaque producer-specific data
    void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE
#endif  // ARROW_FLAG_DICTIONARY_ORDERED

#ifdef __cplusplus
}
#endif

py::array_t<PyObjectGeography> from_geoarrow(py::object input,
                                             bool oriented = false,
                                             bool planar = false) {
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
        // TODO replace with constant
        auto tol = S1Angle::Radians(100.0 / (6371.01 * 1000));
        options.set_tessellate_tolerance(tol);
    }
    reader.Init(schema, options);
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
            If set to True, the edges linestrings and polygons are assumed to
            be planar. In that case, additional points will be added to the line
            while creating the geography objects, to ensure every point is
            within 100m of the original line.
            By default (False), it is assumed that the edges are spherical
            (i.e. represent the shortest path on the sphere between two points).
    )pbdoc");
}
