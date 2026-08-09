#include <pybind11/pybind11.h>
#include <s2/s2boolean_operation.h>
#include <s2geography.h>

#include <unordered_map>

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

enum class PredicateId {
    intersects,
    within,
    contains,
    equals,
    covers,
    covered_by,
    touches,
    disjoint
};

const std::unordered_map<std::string, PredicateId> predicate_ids = {
    {"intersects", PredicateId::intersects},
    {"within", PredicateId::within},
    {"contains", PredicateId::contains},
    {"equals", PredicateId::equals},
    {"covers", PredicateId::covers},
    {"covered_by", PredicateId::covered_by},
    {"touches", PredicateId::touches},
    {"disjoint", PredicateId::disjoint},
};

}  // namespace

PredicateFunc get_predicate(const std::string& name) {
    using Index = s2geog::ShapeIndexGeography;

    auto it = predicate_ids.find(name);
    if (it == predicate_ids.end()) {
        throw py::value_error("invalid predicate: '" + name + "'");
    }

    switch (it->second) {
        case PredicateId::intersects:
            return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
                return s2geog::s2_intersects(a, b, options);
            };
        case PredicateId::within:
            return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
                return s2geog::s2_contains(b, a, options);
            };
        case PredicateId::contains:
            return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
                return s2geog::s2_contains(a, b, options);
            };
        case PredicateId::equals:
            return [options = S2BooleanOperation::Options()](const Index& a, const Index& b) {
                return s2geog::s2_equals(a, b, options);
            };
        case PredicateId::covers:
            return [options = closed_options()](const Index& a, const Index& b) {
                return s2geog::s2_contains(a, b, options);
            };
        case PredicateId::covered_by:
            return [options = closed_options()](const Index& a, const Index& b) {
                return s2geog::s2_contains(b, a, options);
            };
        case PredicateId::touches:
            return [closed = closed_options(), open = open_options()](const Index& a,
                                                                      const Index& b) {
                return s2geog::s2_intersects(a, b, closed) && !s2geog::s2_intersects(a, b, open);
            };
        case PredicateId::disjoint:
            throw py::value_error(
                "the 'disjoint' predicate is not supported by SpatialIndex.query: it is not a "
                "refinement of the spatial-index candidate set");
    }

    throw py::value_error("invalid predicate: '" + name + "'");  // unreachable
}

}  // namespace spherely
