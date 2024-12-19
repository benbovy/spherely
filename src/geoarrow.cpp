#include <s2geography.h>

#include "arrow_abi.h"
#include "constants.hpp"
#include "creation.hpp"
#include "geography.hpp"
#include "projections.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

py::array_t<PyObjectGeography> from_geoarrow(py::object input,
                                             bool oriented,
                                             bool planar,
                                             float tessellate_tolerance,
                                             Projection projection,
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
    options.set_projection(projection.s2_projection());
    if (planar) {
        auto tol = S1Angle::Radians(tessellate_tolerance / numeric_constants::EARTH_RADIUS_METERS);
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

    void set_schema(ArrowSchema* schema_src) {
        memcpy(&schema_, schema_src, sizeof(struct ArrowSchema));
        schema_src->release = nullptr;
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

ArrowArrayHolder to_geoarrow(py::array_t<PyObjectGeography> input,
                             py::object output_schema,
                             Projection projection,
                             bool planar,
                             float tessellate_tolerance,
                             int precision) {
    ArrowArrayHolder array = ArrowArrayHolder();

    s2geog::geoarrow::Writer writer;
    std::vector<std::unique_ptr<s2geog::Geography>> s2geog_vec;

    s2geog::geoarrow::ExportOptions options;
    options.set_precision(precision);
    options.set_projection(projection.s2_projection());
    if (planar) {
        auto tol = S1Angle::Radians(tessellate_tolerance / numeric_constants::EARTH_RADIUS_METERS);
        options.set_tessellate_tolerance(tol);
    }

    if (!output_schema.is(py::none())) {
        if (!py::hasattr(output_schema, "__arrow_c_schema__")) {
            throw std::invalid_argument(
                "'output_schema' should be an Arrow-compatible schema object "
                "(i.e. has an '__arrow_c_schema__' method)");
        }
        py::capsule schema_capsule = output_schema.attr("__arrow_c_schema__")();
        ArrowSchema* schema = static_cast<ArrowSchema*>(schema_capsule);
        try {
            writer.Init(schema, options);
        } catch (const std::exception& ex) {
            // re-raise RuntimeError as ValueError
            if (strlen(ex.what()) > 0) {
                throw py::value_error(ex.what());
            } else {
                throw py::value_error("Error initializing writer. Did you pass a valid schema?");
            }
        }
        array.set_schema(schema);
        // TODO add support for specifying a geometry encoding
        // } else if (geometry_encoding.equal(py::str("WKT"))) {
        //     writer.Init(s2geog::geoarrow::Writer::OutputType::kWKT, options);
        // } else if (geometry_encoding.equal(py::str("WKB"))) {
        //     writer.Init(s2geog::geoarrow::Writer::OutputType::kWKB, options);
    } else {
        throw std::invalid_argument("'output_schema' should be specified");
    }

    for (int i = 0; i < input.size(); i++) {
        writer.WriteGeography((*input.data(i)).as_geog_ptr()->geog());
    }

    writer.Finish(array.array());

    return std::move(array);
}

void init_geoarrow(py::module& m) {
    py::class_<ArrowArrayHolder>(m, "ArrowArrayHolder")
        .def("__arrow_c_array__", &ArrowArrayHolder::return_capsules);

    m.def(
        "from_geoarrow",
        &from_geoarrow,
        py::arg("geographies"),
        py::pos_only(),
        py::kw_only(),
        py::arg("oriented") = false,
        py::arg("planar") = false,
        py::arg("tessellate_tolerance") = 100.0,
        py::arg("projection") = Projection::lnglat(),
        py::arg("geometry_encoding") = py::none(),
        R"pbdoc(from_geoarrow(geographies, /, *, oriented=False, planar=False, tessellate_tolerance=100.0, projection=spherely.Projection.lnglat(), geometry_encoding=None)

        Create an array of geographies from an Arrow array object with a GeoArrow
        extension type.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        This functions accepts any Arrow array object implementing
        the `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_array__``
        method).

        .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

        Parameters
        ----------
        geographies : pyarrow.Array, Arrow array
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
            ensure every point is within the `tessellate_tolerance` distance
            (100m by default) of the original line.
            By default (False), it is assumed that the edges are spherical
            (i.e. represent the shortest path on the sphere between two points).
        tessellate_tolerance : float, default 100.0
            The maximum distance in meters that a point must be moved to
            satisfy the planar edge constraint. This is only used if `planar`
            is set to True.
        projection : spherely.Projection, default Projection.lnglat()
            The projection of the input coordinates. By default, it assumes
            longitude/latitude coordinates, but this option allows to convert
            from coordinates in pseudo-mercator or orthographic projection as well.
        geometry_encoding : str, default None
            By default, the encoding is inferred from the GeoArrow extension
            type of the input array.
            However, for parsing WKT and WKB it is also possible to pass an
            Arrow array without geoarrow type but with a plain string or
            binary type, if specifying this keyword with "WKT" or "WKB",
            respectively.

        Returns
        -------
        Geography or array
            An array of geography objects.

    )pbdoc");

    m.def(
        "to_geoarrow",
        &to_geoarrow,
        py::arg("geographies"),
        py::pos_only(),
        py::kw_only(),
        py::arg("output_schema") = py::none(),
        py::arg("projection") = Projection::lnglat(),
        py::arg("planar") = false,
        py::arg("tessellate_tolerance") = 100.0,
        py::arg("precision") = 6,
        R"pbdoc(to_geoarrow(geographies, /, *, output_schema=None, projection=spherely.Projection.lnglat(), planar=False, tessellate_tolerance=100.0, precision=6)

        Convert an array of geographies to an Arrow array object with a GeoArrow
        extension type.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        Parameters
        ----------
        geographies : array_like
            An array of :py:class:`~spherely.Geography` objects.
        output_schema : Arrow schema, pyarrow.DataType, pyarrow.Field, default None
            The geoarrow extension type to use for the output. This can indicate
            one of the native geoarrow types (e.g. "point", "linestring", "polygon",
            etc) or the serialized WKT or WKB options.
            The type can be specified with any Arrow schema compatible object
            (any object implementing the Arrow PyCapsule Protocol for schemas,
            i.e. which has a ``__arrow_c_schema__`` method). For example, this
            can be a ``pyarrow.DataType`` or ``pyarrow.Field``, and you can
            use the ``geoarrow.pyarrow`` package to construct such geoarrow
            extension types.
        projection : spherely.Projection, default Projection.lnglat()
            The projection to use when converting the geographies to the output
            encoding. By default, uses longitude/latitude coordinates ("plate
            carree" projection).
        planar : bool, default False
            If set to True, the edges of linestrings and polygons in the output
            are assumed to be linear on the plane. In that case, additional
            points will be added to the line while converting to the output
            encoding, to ensure every point is within the `tessellate_tolerance`
            distance (100m by default) of the original line on the sphere.
        tessellate_tolerance : float, default 100.0
            The maximum distance in meters that a point must be moved to
            satisfy the planar edge constraint. This is only used if `planar`
            is set to True.
        precision : int, default 6
            The number of decimal places to include in the output. Only used
            when writing as WKT.

        Returns
        -------
        ArrowArrayHolder
            A generic Arrow array object with geographies encoded to GeoArrow.

        Examples
        --------
        >>> import spherely
        >>> import geoarrow.pyarrow as ga
        >>> arr = spherely.to_geoarrow(arr, output_schema=ga.point())

        The returned object is a generic Arrow-compatible object that then
        can be consumed by your Arrow library of choice. For example, using
        ``pyarrow``:

        >>> import pyarrow as pa
        >>> arr_pa = pa.array(arr)

    )pbdoc");
}
