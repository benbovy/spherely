#include <pybind11/pybind11.h>
#include <s2/s2boolean_operation.h>
#include <s2geography.h>

#include "predicates.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;

namespace spherely {

namespace {

S2BooleanOperation::Options closed_options() {
    S2BooleanOperation::Options options;
    options.set_polyline_model(S2BooleanOperation::PolylineModel::CLOSED);
    options.set_polygon_model(S2BooleanOperation::PolygonModel::CLOSED);
    return options;
}

S2BooleanOperation::Options open_options() {
    S2BooleanOperation::Options options;
    options.set_polyline_model(S2BooleanOperation::PolylineModel::OPEN);
    options.set_polygon_model(S2BooleanOperation::PolygonModel::OPEN);
    return options;
}

}  // namespace

PredicateFunc get_predicate(const std::string& name) {
    using Index = s2geog::ShapeIndexGeography;

    if (name == "intersects") {
        return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
            return s2geog::s2_intersects(a, b, options);
        };
    } else if (name == "within") {
        return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
            return s2geog::s2_contains(b, a, options);
        };
    } else if (name == "contains") {
        return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
            return s2geog::s2_contains(a, b, options);
        };
    } else if (name == "equals") {
        return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
            return s2geog::s2_equals(a, b, options);
        };
    } else if (name == "covers") {
        return [options = closed_options()](const Index& a, const Index& b) {
            return s2geog::s2_contains(a, b, options);
        };
    } else if (name == "covered_by") {
        return [options = closed_options()](const Index& a, const Index& b) {
            return s2geog::s2_contains(b, a, options);
        };
    } else if (name == "touches") {
        return [closed = closed_options(), open = open_options()](const Index& a, const Index& b) {
            return s2geog::s2_intersects(a, b, closed) && !s2geog::s2_intersects(a, b, open);
        };
    } else if (name == "disjoint") {
        throw py::value_error(
            "the 'disjoint' predicate is not supported by SpatialIndex.query: it is not a "
            "refinement of the spatial-index candidate set");
    } else {
        throw py::value_error("invalid predicate: '" + name + "'");
    }
}

}  // namespace spherely
