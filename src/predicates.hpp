#ifndef SPHERELY_PREDICATES_H_
#define SPHERELY_PREDICATES_H_

#include <s2geography.h>

#include <functional>
#include <string>

namespace spherely {

namespace s2geog = s2geography;

// A binary spatial predicate operating on two indexed geographies, i.e.
// ``predicate(a, b)`` where ``a`` and ``b`` are ``ShapeIndexGeography``.
using PredicateFunc =
    std::function<bool(const s2geog::ShapeIndexGeography&, const s2geog::ShapeIndexGeography&)>;

// Returns the predicate closure matching ``name`` (one of "intersects",
// "within", "contains", "covers", "covered_by", "touches", "equals").
//
// Throws ``pybind11::value_error`` for unknown names and for "disjoint" (which
// cannot be evaluated as a refinement of a spatial-index candidate set).
//
// The closures mirror the semantics of the vectorized predicates registered in
// ``predicates.cpp``; that file remains the reference implementation.
PredicateFunc get_predicate(const std::string& name);

}  // namespace spherely

#endif  // SPHERELY_PREDICATES_H_
