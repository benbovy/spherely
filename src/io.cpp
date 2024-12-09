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

    PyObjectGeography operator()(py::str a) const {
        return make_py_geography(m_reader->read_feature(a));
    }

private:
    std::shared_ptr<s2geog::WKTReader> m_reader;
};

class ToWKT {
public:
    ToWKT(int precision = 6) {
        m_writer = std::make_shared<s2geog::WKTWriter>(precision);
    }

    py::str operator()(PyObjectGeography a) const {
        auto res = m_writer->write_feature(a.as_geog_ptr()->geog());
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

    PyObjectGeography operator()(py::bytes a) const {
        return make_py_geography(m_reader->ReadFeature(a));
    }

private:
    std::shared_ptr<s2geog::WKBReader> m_reader;
};

class ToWKB {
public:
    ToWKB() {
        m_writer = std::make_shared<s2geog::WKBWriter>();
    }

    py::bytes operator()(PyObjectGeography a) const {
        return m_writer->WriteFeature(a.as_geog_ptr()->geog());
    }

private:
    std::shared_ptr<s2geog::WKBWriter> m_writer;
};

void init_io(py::module& m) {
    m.def(
        "from_wkt",
        [](py::array_t<py::str> a, bool oriented, bool planar, float tessellate_tolerance) {
            return py::vectorize(FromWKT(oriented, planar, tessellate_tolerance))(std::move(a));
        },
        py::arg("a"),
        py::arg("oriented") = false,
        py::arg("planar") = false,
        py::arg("tessellate_tolerance") = 100.0,
        R"pbdoc(
        Creates geographies from the Well-Known Text (WKT) representation.

        Parameters
        ----------
        a : str or array_like
            WKT strings.
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

    )pbdoc");

    m.def(
        "to_wkt",
        [](py::array_t<PyObjectGeography> a, int precision) {
            return py::vectorize(ToWKT(precision))(std::move(a));
        },
        py::arg("a"),
        py::arg("precision") = 6,
        R"pbdoc(
        Returns the WKT representation of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object(s)
        precision : int, default 6
            The number of decimal places to include in the output.

    )pbdoc");

    m.def(
        "from_wkb",
        [](py::array_t<py::bytes> a, bool oriented, bool planar, float tessellate_tolerance) {
            return py::vectorize(FromWKB(oriented, planar, tessellate_tolerance))(std::move(a));
        },
        py::arg("a"),
        py::arg("oriented") = false,
        py::arg("planar") = false,
        py::arg("tessellate_tolerance") = 100.0,
        R"pbdoc(
        Creates geographies from the Well-Known Bytes (WKB) representation.

        Parameters
        ----------
        a : bytes or array_like
            WKB objects.
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

    )pbdoc");

    m.def("to_wkb",
          py::vectorize(ToWKB()),
          py::arg("a"),
          R"pbdoc(
        Returns the WKB representation of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");
}
