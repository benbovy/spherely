#include <s2geography.h>

#include "arrow_abi.h"
#include "constants.hpp"
#include "creation.hpp"
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
    if (!py::hasattr(input, "__arrow_c_array__")) {
        throw std::invalid_argument(
            "input should be an Arrow-compatible array object (i.e. has an '__arrow_c_array__' "
            "method)");
    }
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
        try {
            reader.Init(schema, options);
        } catch (const std::exception& ex) {
            // re-raise RuntimeError as ValueError
            throw py::value_error(ex.what());
        }
    } else if (geometry_encoding.equal(py::str("WKT"))) {
        reader.Init(s2geog::geoarrow::Reader::InputType::kWKT, options);
    } else if (geometry_encoding.equal(py::str("WKB"))) {
        reader.Init(s2geog::geoarrow::Reader::InputType::kWKB, options);
    } else {
        throw std::invalid_argument("'geometry_encoding' should be one of None, 'WKT' or 'WKB'");
    }

    try {
        reader.ReadGeography(array, 0, array->length, &s2geog_vec);
    } catch (const std::exception& ex) {
        // re-raise RuntimeError as ValueError
        throw py::value_error(ex.what());
    }

    // Convert resulting vector to array of python objects
    auto result = py::array_t<PyObjectGeography>(array->length);
    py::buffer_info rbuf = result.request();
    py::object* rptr = static_cast<py::object*>(rbuf.ptr);

    py::ssize_t i = 0;
    for (auto& s2geog_ptr : s2geog_vec) {
        rptr[i] = make_py_geography(std::move(s2geog_ptr));
        i++;
    }
    return result;
}

/// \brief Object holding (and managing the memory) of an Arrow array (ArrowArray and ArraySchema
/// combo)
class ArrowArrayHolder {
public:
    /// \brief Construct an invalid instance holding no resources
    ArrowArrayHolder() {
        array_.release = nullptr;
        schema_.release = nullptr;
    }

    /// \brief Move and take ownership of data
    ArrowArrayHolder(ArrowArray* array, ArrowSchema* schema) {
        move(array, schema, &array_, &schema_);
    }

    /// \brief Move and take ownership of data wrapped by rhs
    ArrowArrayHolder(ArrowArrayHolder&& rhs) : ArrowArrayHolder(rhs.array(), rhs.schema()) {}
    ArrowArrayHolder& operator=(ArrowArrayHolder&& rhs) {
        reset(rhs.array(), rhs.schema());
        return *this;
    }

    // These objects are not copyable
    ArrowArrayHolder(const ArrowArrayHolder& rhs) = delete;

    /// \brief Get a pointer to the data owned by this object
    ArrowArray* array() noexcept {
        return &array_;
    }
    const ArrowArray* array() const noexcept {
        return &array_;
    }

    ArrowSchema* schema() noexcept {
        return &schema_;
    }
    const ArrowSchema* schema() const noexcept {
        return &schema_;
    }

    py::tuple return_capsules(py::args args, const py::kwargs& kwargs) {
        if ((args.size() > 0) && (!args[0].is_none())) {
            throw std::invalid_argument(
                "Additional arguments (such as requested_schema) with a non-default value are not "
                "supported");
        }
        if (kwargs) {
            for (auto& item : kwargs) {
                if (!item.second.is_none()) {
                    throw std::invalid_argument(
                        "Additional arguments (such as requested_schema) with a non-default value "
                        "are not supported");
                }
            }
        }

        ArrowArray* c_array = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray)));
        ArrowSchema* c_schema = static_cast<ArrowSchema*>(malloc(sizeof(ArrowSchema)));
        move(&array_, &schema_, c_array, c_schema);

        constexpr auto array_cleanup = [](void* ptr) noexcept {
            auto array = static_cast<ArrowArray*>(ptr);
            if (array->release != nullptr) {
                array->release(array);
            }
            free(array);
        };
        py::capsule array_capsule{c_array, "arrow_array", array_cleanup};

        constexpr auto schema_cleanup = [](void* ptr) noexcept {
            auto schema = static_cast<ArrowSchema*>(ptr);
            if (schema->release != nullptr) {
                schema->release(schema);
            }
            free(schema);
        };
        py::capsule schema_capsule{c_schema, "arrow_schema", schema_cleanup};

        return py::make_tuple(schema_capsule, array_capsule);
    }

    void move(ArrowArray* array_src,
              ArrowSchema* schema_src,
              ArrowArray* array_dst,
              ArrowSchema* schema_dst) {
        memcpy(array_dst, array_src, sizeof(struct ArrowArray));
        array_src->release = nullptr;

        memcpy(schema_dst, schema_src, sizeof(struct ArrowSchema));
        schema_src->release = nullptr;
    }

    /// \brief Call data's release callback if valid
    void reset() {
        if (array_.release != nullptr) {
            array_.release(&array_);
        }

        if (schema_.release != nullptr) {
            schema_.release(&schema_);
        }
    }

    /// \brief Call data's release callback if valid and move ownership of the data
    /// pointed to by data
    void reset(ArrowArray* array_src, ArrowSchema* schema_src) {
        reset();
        move(array_src, schema_src, &array_, &schema_);
    }

    ~ArrowArrayHolder() {
        reset();
    }

protected:
    ArrowArray array_;
    ArrowSchema schema_;
};

ArrowArrayHolder to_geoarrow(py::array_t<PyObjectGeography> input, py::object geometry_encoding) {
    ArrowArrayHolder array = ArrowArrayHolder();

    s2geog::geoarrow::Writer writer;
    std::vector<std::unique_ptr<s2geog::Geography>> s2geog_vec;

    s2geog::geoarrow::ImportOptions options;
    // TODO replace with constant
    auto tol = S1Angle::Radians(100.0 / (6371.01 * 1000));
    options.set_tessellate_tolerance(tol);
    options.set_projection(s2geog::geoarrow::mercator());

    if (geometry_encoding.is(py::none())) {
        // writer.Init(schema, options);
        throw std::invalid_argument("not yet implemented");
    } else if (geometry_encoding.equal(py::str("points"))) {
        writer.Init(s2geog::geoarrow::Writer::OutputType::kPoints, options, array.schema());
    } else if (geometry_encoding.equal(py::str("WKT"))) {
        writer.Init(s2geog::geoarrow::Writer::OutputType::kWKT, options, array.schema());
    } else if (geometry_encoding.equal(py::str("WKB"))) {
        writer.Init(s2geog::geoarrow::Writer::OutputType::kWKB, options, array.schema());
    } else {
        throw std::invalid_argument("'geometry_encoding' should be one of None, 'WKT' or 'WKB'");
    }

    size_t num_geographies = static_cast<size_t>(input.size());

    const s2geog::Geography** geographies = static_cast<const s2geog::Geography**>(
        malloc(sizeof(const s2geog::Geography*) * num_geographies));

    for (int i = 0; i < input.size(); i++) {
        writer.WriteGeography((*input.data(i)).as_geog_ptr()->geog());
    }

    writer.Finish(array.array());

    return std::move(array);
}

void init_geoarrow(py::module& m) {
    py::class_<ArrowArrayHolder>(m, "ArrowArrayHolder")
        .def("__arrow_c_array__", &ArrowArrayHolder::return_capsules);

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

    m.def("to_geoarrow",
          &to_geoarrow,
          py::arg("input"),
          py::pos_only(),
          py::kw_only(),
          py::arg("geometry_encoding") = py::none(),
          R"pbdoc(
        Convert an array of geographies to an Arrow array object with a GeoArrow
        extension type.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        Parameters
        ----------
        input : array_like
            An array of geography objects.
        geometry_encoding : str, default None
            By default, the encoding is inferred from the GeoArrow extension
            type of the input array.
            However, for serializing to WKT and WKB it is also possible to pass
    )pbdoc");
}
