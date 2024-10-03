#include <s2geography.h>

#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

// PyObjectGeography from_wkt(std::string a) {
//     s2geog::WKTReader reader;
//     std::unique_ptr<s2geog::Geography> s2geog = reader.read_feature(a);
//     auto geog_ptr = std::make_unique<spherely::Geography>(std::move(s2geog));
//     return PyObjectGeography::from_geog(std::move(geog_ptr));
// }

// void init_geoarrow(py::module& m) {
//     m.def("from_wkt",
//           py::vectorize(&from_wkt),
//           py::arg("a"),
//           R"pbdoc(
//         Creates a geography object from a WKT string.

//         Parameters
//         ----------
//         a : str
//             WKT string

//     )pbdoc");
// }

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

py::array_t<PyObjectGeography> from_geoarrow(py::object input) {
    py::tuple capsules = input.attr("__arrow_c_array__")();
    py::capsule schema_capsule = capsules[0];
    py::capsule array_capsule = capsules[1];

    const ArrowSchema* schema = static_cast<const ArrowSchema*>(schema_capsule);
    const ArrowArray* array = static_cast<const ArrowArray*>(array_capsule);

    s2geog::geoarrow::Reader reader;
    std::vector<std::unique_ptr<s2geog::Geography>> s2geog_vec;

    reader.Init(schema, s2geog::geoarrow::ImportOptions());
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
    m.def("from_geoarrow", &from_geoarrow);
}
