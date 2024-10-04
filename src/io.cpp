#include <s2/s1angle.h>
#include <s2geography.h>

#include "constants.hpp"
#include "geography.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

PyObjectGeography from_wkt(py::str a, bool oriented, bool planar) {
#if defined(S2GEOGRAPHY_VERSION_MAJOR) && \
    (S2GEOGRAPHY_VERSION_MAJOR >= 1 || S2GEOGRAPHY_VERSION_MINOR >= 2)
    s2geog::geoarrow::ImportOptions options;
    options.set_oriented(oriented);
    if (planar) {
        auto tol = S1Angle::Radians(100.0 / EARTH_RADIUS_METERS);
        options.set_tessellate_tolerance(tol);
    }
    s2geog::WKTReader reader(options);
#else
    if (planar || oriented) {
        throw std::invalid_argument(
            "planar and oriented options are only available with s2geography >= 0.2");
    }
    s2geog::WKTReader reader;
#endif
    std::unique_ptr<s2geog::Geography> s2geog = reader.read_feature(a);
    auto geog_ptr = std::make_unique<spherely::Geography>(std::move(s2geog));
    return PyObjectGeography::from_geog(std::move(geog_ptr));
}

py::str to_wkt(PyObjectGeography a) {
    s2geog::WKTWriter writer;
    auto res = writer.write_feature(a.as_geog_ptr()->geog());
    return py::str(res);
}

void init_io(py::module& m) {
    m.def("from_wkt",
          py::vectorize(&from_wkt),
          py::arg("a"),
          py::arg("oriented") = false,
          py::arg("planar") = false,
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
            If set to True, the edges linestrings and polygons are assumed to
            be planar. In that case, additional points will be added to the line
            while creating the geography objects, to ensure every point is
            within 100m of the original line.
            By default (False), it is assumed that the edges are spherical
            (i.e. represent the shortest path on the sphere between two points).

    )pbdoc");

    m.def("to_wkt",
          py::vectorize(&to_wkt),
          py::arg("a"),
          R"pbdoc(
        Returns the WKT representation of each geography.

        Parameters
        ----------
        a : :py:class:`Geography` or array_like
            Geography object(s)

    )pbdoc");
}
