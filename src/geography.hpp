#ifndef SPHERELY_GEOGRAPHY_H_
#define SPHERELY_GEOGRAPHY_H_

#include <s2geography/geography.h>

#include <memory>
#include <type_traits>

#include "s2/s2lax_polyline_shape.h"
#include "s2/s2loop.h"
#include "s2/s2point.h"
#include "s2/s2polyline.h"
#include "s2geography.h"

namespace s2geog = s2geography;

namespace spherely {

using S2GeographyPtr = std::unique_ptr<s2geog::Geography>;
using S2GeographyIndexPtr = std::unique_ptr<s2geog::ShapeIndexGeography>;

/*
** The registered Geography types
*/
enum class GeographyType : std::int8_t {
    None = -1,
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeographyCollection
};

/*
** Thin wrapper around s2geography::Geography.
**
** This wrapper implements the following specific features (that might
** eventually move into s2geography::Geography?):
**
** - Implement move semantics only.
** - add ``clone()`` method for explicit copy
** - Add ``geog_type()`` method for getting the geography type
** - Eagerly infer the geography type as well as other properties
** - Encapsulate a lazy ``s2geography::ShapeIndexGeography`` accessible via ``geog_index()``.
**
*/
class Geography {
public:
    Geography(const Geography&) = delete;
    Geography(Geography&& geog) : m_s2geog_ptr(std::move(geog.m_s2geog_ptr)) {
        extract_geog_properties();
    }
    Geography(S2GeographyPtr&& s2geog_ptr) : m_s2geog_ptr(std::move(s2geog_ptr)) {
        extract_geog_properties();
    }

    Geography& operator=(const Geography&) = delete;
    Geography& operator=(Geography&& other) {
        m_s2geog_ptr = std::move(other.m_s2geog_ptr);
        m_is_empty = other.m_is_empty;
        m_geog_type = other.m_geog_type;
        return *this;
    }

    inline GeographyType geog_type() const noexcept {
        return m_geog_type;
    }

    inline const s2geog::Geography& geog() const noexcept {
        return *m_s2geog_ptr;
    }

    template <class T, std::enable_if_t<std::is_base_of_v<s2geog::Geography, T>, bool> = true>
    inline const T* downcast_geog() const {
        return dynamic_cast<const T*>(&geog());
    }

    inline const s2geog::ShapeIndexGeography& geog_index() {
        if (!m_s2geog_index_ptr) {
            m_s2geog_index_ptr = std::make_unique<s2geog::ShapeIndexGeography>(geog());
        }

        return *m_s2geog_index_ptr;
    }
    inline void reset_index() {
        m_s2geog_index_ptr.reset();
    }
    inline bool has_index() const noexcept {
        return m_s2geog_index_ptr != nullptr;
    }

    inline int dimension() const {
        return m_s2geog_ptr->dimension();
    }
    inline int num_shapes() const {
        return m_s2geog_ptr->num_shapes();
    }
    inline bool is_empty() const noexcept {
        return m_is_empty;
    }

    Geography clone() const;

private:
    S2GeographyPtr m_s2geog_ptr;
    S2GeographyIndexPtr m_s2geog_index_ptr;
    bool m_is_empty = false;
    GeographyType m_geog_type;

    // We don't want Geography to be default constructible, except internally via `clone()`
    // where there is no need to infer geography properties as we already know them.
    Geography() : m_is_empty(true) {}

    void extract_geog_properties();
};

}  // namespace spherely

#endif  // SPHERELY_GEOGRAPHY_H_
