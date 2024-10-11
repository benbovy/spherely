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
** Wrap one or more s2geometry objects into a spherely::Geography object.
**
** @tparam T The s2geography type
** @tparam S The type of the s2geometry (vector of) object(s)
** @param s2_obj A single or a vector of s2geometry objects (e.g., S2Point, S2Polyline, etc.)
** @returns A new spherely::Geography object
*/
template <class T, class S, std::enable_if_t<std::is_base_of_v<s2geog::Geography, T>, bool> = true>
inline std::unique_ptr<Geography> make_geography(S &&s2_obj) {
    auto s2geog_ptr = std::make_unique<T>(std::forward<S>(s2_obj));
    return std::make_unique<Geography>(std::move(s2geog_ptr));
}

/*
** Wrap a s2geography::Geography object into a spherely::Geography object.
**
** @tparam T The s2geography type
** @param s2geog_ptr a pointer to the s2geography::Geography object
** @returns A new spherely::Geography object
*/
template <class T>
inline std::unique_ptr<Geography> make_geography(std::unique_ptr<T> s2geog_ptr) {
    return std::make_unique<Geography>(std::move(s2geog_ptr));
}

/*
** Helper to create a Spherely Python Geography object directly from one or more
** S2Geometry objects.
*
** @tparam T The s2geography type
** @tparam S The type of the s2geometry (vector of) object(s)
** @param s2_obj A single or a vector of s2geometry objects (e.g., S2Point, S2Polyline, etc.)
** @returns A new Python Geography object (pybind11::object)
*/
template <class T, class S, std::enable_if_t<std::is_base_of_v<s2geog::Geography, T>, bool> = true>
inline PyObjectGeography make_py_geography(S &&s2_obj) {
    auto geog_ptr = make_geography<T>(std::forward<S>(s2_obj));
    return PyObjectGeography::from_geog(std::move(geog_ptr));
}

/*
** Helper to create a shperely::Geography object from one s2geography::Geography
** object.
*
** @tparam T The S2Geography type
** @param s2geog_ptr a pointer to the s2geography::Geography object
** @returns A new Python Geography object (pybind11::object)
*/
template <class T>
inline PyObjectGeography make_py_geography(std::unique_ptr<T> s2geog_ptr) {
    auto geog_ptr = make_geography(std::move(s2geog_ptr));
    return PyObjectGeography::from_geog(std::move(geog_ptr));
}

}  // namespace spherely

#endif  // SPHERELY_CREATION_H_
