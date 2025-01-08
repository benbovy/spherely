#include <s2/s1angle.h>
#include <s2geography.h>

#include "constants.hpp"
#include "creation.hpp"
#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

class FromWKT {
public:
    FromWKT(bool oriented, bool planar, float tessellate_tolerance = 100.0) {
        s2geog::geoarrow::ImportOptions options;
        options.set_oriented(oriented);
        if (planar) {
            auto tol =
                S1Angle::Radians(tessellate_tolerance / numeric_constants::EARTH_RADIUS_METERS);
            options.set_tessellate_tolerance(tol);
        }
        m_reader = std::make_shared<s2geog::WKTReader>(options);
    }

    PyObjectGeography operator()(py::str string) const {
        return make_py_geography(m_reader->read_feature(string));
    }

private:
    std::shared_ptr<s2geog::WKTReader> m_reader;
};

class ToWKT {
public:
    ToWKT(int precision = 6) {
        m_writer = std::make_shared<s2geog::WKTWriter>(precision);
    }

    py::str operator()(PyObjectGeography obj) const {
        auto res = m_writer->write_feature(obj.as_geog_ptr()->geog());
        return py::str(res);
    }

private:
    std::shared_ptr<s2geog::WKTWriter> m_writer;
};

class FromWKB {
public:
    FromWKB(bool oriented, bool planar, float tessellate_tolerance = 100.0) {
        s2geog::geoarrow::ImportOptions options;
        options.set_oriented(oriented);
        if (planar) {
            auto tol =
                S1Angle::Radians(tessellate_tolerance / numeric_constants::EARTH_RADIUS_METERS);
            options.set_tessellate_tolerance(tol);
        }
        m_reader = std::make_shared<s2geog::WKBReader>(options);
    }

    PyObjectGeography operator()(py::bytes bytes) const {
        return make_py_geography(m_reader->ReadFeature(bytes));
    }

private:
    std::shared_ptr<s2geog::WKBReader> m_reader;
};

class ToWKB {
public:
    ToWKB() {
        m_writer = std::make_shared<s2geog::WKBWriter>();
    }

    py::bytes operator()(PyObjectGeography obj) const {
        return m_writer->WriteFeature(obj.as_geog_ptr()->geog());
    }

private:
    std::shared_ptr<s2geog::WKBWriter> m_writer;
};

void init_io(py::module& m) {
    m.def(
        "from_wkt",
        [](py::array_t<py::str> string, bool oriented, bool planar, float tessellate_tolerance) {
            return py::vectorize(FromWKT(oriented, planar, tessellate_tolerance))(
                std::move(string));
        },
        py::arg("geography"),
        py::pos_only(),
        py::kw_only(),
        py::arg("oriented") = false,
        py::arg("planar") = false,
        py::arg("tessellate_tolerance") = 100.0,
        R"pbdoc(from_wkt(geography, /, *, oriented=False, planar=False, tessellate_tolerance=100.0)

        Creates geographies from the Well-Known Text (WKT) representation.

        Parameters
        ----------
        geography : str or array_like
            The WKT string(s) to convert.
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

        Returns
        -------
        Geography or array
            A single or an array of geography objects.

    )pbdoc");

    m.def(
        "to_wkt",
        [](py::array_t<PyObjectGeography> obj, int precision) {
            return py::vectorize(ToWKT(precision))(std::move(obj));
        },
        py::arg("geography"),
        py::pos_only(),
        py::arg("precision") = 6,
        R"pbdoc(to_wkt(geography, /, precision=6)

        Returns the WKT representation of each geography.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).
        precision : int, default 6
            The number of decimal places to include in the output.

        Returns
        -------
        str or array
            A string or an array of strings.

    )pbdoc");

    m.def(
        "from_wkb",
        [](py::array_t<py::bytes> bytes, bool oriented, bool planar, float tessellate_tolerance) {
            return py::vectorize(FromWKB(oriented, planar, tessellate_tolerance))(std::move(bytes));
        },
        py::arg("geography"),
        py::pos_only(),
        py::kw_only(),
        py::arg("oriented") = false,
        py::arg("planar") = false,
        py::arg("tessellate_tolerance") = 100.0,
        R"pbdoc(from_wkb(geography, /, *, oriented=False, planar=False, tessellate_tolerance=100.0)

        Creates geographies from the Well-Known Bytes (WKB) representation.

        Parameters
        ----------
        geography : bytes or array_like
            The WKB byte object(s) to convert.
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

        Returns
        -------
        Geography or array
            A single or an array of geography objects.

    )pbdoc");

    m.def("to_wkb",
          py::vectorize(ToWKB()),
          py::arg("geography"),
          py::pos_only(),
          R"pbdoc(to_wkb(geography, /)

        Returns the WKB representation of each geography.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            Geography object(s).

        Returns
        -------
        bytes or array
            A bytes object or an array of bytes.

    )pbdoc");
}
