#ifndef SPHERELY_CREATION_H_
#define SPHERELY_CREATION_H_

#include <memory>
#include <type_traits>

#include "geography.hpp"
#include "pybind11.hpp"

namespace spherely {

//
// ---- S2geometry / S2Geography / Spherely object wrapper utility functions
//

/*
** Helper to wrap a S2Geometry object as a Spherely C++ Geography object.
**
** There are two nested wrap levels:
** spherely::Geography > s2geography::Geography > s2geometry object
**
** @tparam T The S2Geography type
** @tparam S The S2Geometry type
*/
template <class T, class S, std::enable_if_t<std::is_base_of_v<s2geog::Geography, T>, bool> = true>
inline std::unique_ptr<Geography> make_geography(S &&s2_obj) {
    auto s2geog_ptr = std::make_unique<T>(std::forward<S>(s2_obj));
    return std::make_unique<Geography>(std::move(s2geog_ptr));
}

/*
** Helper to create Spherely Python Geography objects directly from S2Geometry objects.
*
** @tparam T The S2Geography type
** @tparam S The S2Geometry type
*/
template <class T, class S, std::enable_if_t<std::is_base_of_v<s2geog::Geography, T>, bool> = true>
inline PyObjectGeography make_py_geography(S &&s2_obj) {
    auto geog = make_geography<T>(std::forward<S>(s2_obj));
    return PyObjectGeography::from_geog(std::move(geog));
}

// Helpers for explicit copy of s2geography objects.
std::unique_ptr<s2geog::Geography> clone_s2geography(const s2geog::Geography &geog);
}  // namespace spherely

#endif  // SPHERELY_CREATION_H_
