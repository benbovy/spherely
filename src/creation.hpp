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
** Helper to wrap one or more S2Geometry objects as a Spherely C++ Geography object.
**
** There are two nested wrap levels:
** spherely::Geography > s2geography::Geography > s2geometry object(s)
**
** @tparam T The S2Geography type
** @tparam S The type of the S2Geometry object(s) passed to the S2Geography constructor
*/
template <class T, class S, std::enable_if_t<std::is_base_of_v<s2geog::Geography, T>, bool> = true>
inline std::unique_ptr<Geography> make_geography(S &&s2_obj) {
    auto s2geog_ptr = std::make_unique<T>(std::forward<S>(s2_obj));
    return std::make_unique<Geography>(std::move(s2geog_ptr));
}

/*
** Helper to create a Spherely Python Geography object directly from one or more
** S2Geometry objects.
*
** @tparam T The S2Geography type
** @tparam S The type of the S2Geometry object(s) passed to the S2Geography constructor
*/
template <class T, class S, std::enable_if_t<std::is_base_of_v<s2geog::Geography, T>, bool> = true>
inline PyObjectGeography make_py_geography(S &&s2_obj) {
    auto geog = make_geography<T>(std::forward<S>(s2_obj));
    return PyObjectGeography::from_geog(std::move(geog));
}

}  // namespace spherely

#endif  // SPHERELY_CREATION_H_
