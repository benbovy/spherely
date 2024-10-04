#ifndef SPHERELY_GEOGRAPHY_H_
#define SPHERELY_GEOGRAPHY_H_

#include <s2geography/geography.h>

#include <memory>

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
** - Add a virtual ``geog_type()`` method for getting the geography type.
** - Encapsulate a lazy ``s2geography::ShapeIndexGeography`` accessible via ``geog_index()``.
**
*/
class Geography {
public:
    using S2GeographyType = s2geog::Geography;

    Geography(const Geography&) = delete;
    Geography(Geography&& geog) : m_s2geog_ptr(std::move(geog.m_s2geog_ptr)) {
        extract_geog_properties();
    }
    Geography(S2GeographyPtr&& s2geog_ptr) : m_s2geog_ptr(std::move(s2geog_ptr)) {
        extract_geog_properties();
    }

    virtual ~Geography() {}

    Geography& operator=(const Geography&) = delete;
    Geography& operator=(Geography&& other) {
        m_s2geog_ptr = std::move(other.m_s2geog_ptr);
        m_is_empty = other.m_is_empty;
        m_geog_type = other.m_geog_type;
        return *this;
    }

    inline GeographyType geog_type() const {
        return m_geog_type;
    }

    inline const s2geog::Geography& geog() const {
        return *m_s2geog_ptr;
    }

    inline const s2geog::ShapeIndexGeography& geog_index() {
        if (!m_s2geog_index_ptr) {
            m_s2geog_index_ptr = std::make_unique<s2geog::ShapeIndexGeography>(geog());
        }

        return *m_s2geog_index_ptr;
    }
    void reset_index() {
        m_s2geog_index_ptr.reset();
    }
    bool has_index() {
        return m_s2geog_index_ptr != nullptr;
    }

    int dimension() const {
        return m_s2geog_ptr->dimension();
    }
    int num_shapes() const {
        return m_s2geog_ptr->num_shapes();
    }
    bool is_empty() const {
        return m_is_empty;
    }

private:
    S2GeographyPtr m_s2geog_ptr;
    S2GeographyIndexPtr m_s2geog_index_ptr;
    bool m_is_empty = false;
    GeographyType m_geog_type;

    template <class T>
    const T* downcast_geog() const {
        return dynamic_cast<const T*>(&geog());
    }

    void extract_geog_properties() {
        if (const auto* ptr = downcast_geog<s2geog::PointGeography>(); ptr) {
            if (ptr->Points().empty()) {
                m_is_empty = true;
            }
            if (ptr->Points().size() <= 1) {
                m_geog_type = GeographyType::Point;
            } else {
                m_geog_type = GeographyType::MultiPoint;
            }
        } else if (const auto* ptr = downcast_geog<s2geog::PolylineGeography>(); ptr) {
            if (ptr->Polylines().empty()) {
                m_is_empty = true;
            }
            if (ptr->Polylines().size() <= 1) {
                m_geog_type = GeographyType::LineString;
            } else {
                m_geog_type = GeographyType::MultiLineString;
            }
        } else if (const auto* ptr = downcast_geog<s2geog::PolygonGeography>(); ptr) {
            int nloops = ptr->Polygon()->num_loops();
            if (nloops == 0) {
                m_is_empty = 0;
            }
            if (nloops <= 1) {
                m_geog_type = GeographyType::Polygon;
            } else {
                m_geog_type = GeographyType::MultiPolygon;
            }
        } else if (const auto* ptr = downcast_geog<s2geog::GeographyCollection>(); ptr) {
            if (ptr->Features().empty()) {
                m_is_empty = 0;
            }
            m_geog_type = GeographyType::GeographyCollection;
        } else {
            m_geog_type = GeographyType::None;
        }
    }
};

class Point : public Geography {
public:
    using S2GeographyType = s2geog::PointGeography;

    Point(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

    inline GeographyType geog_type() const {
        return GeographyType::Point;
    }

    inline const S2Point& s2point() const {
        const auto& points = static_cast<const s2geog::PointGeography&>(geog()).Points();
        // TODO: does not work for empty point geography
        return points[0];
    }
};

class MultiPoint : public Geography {
public:
    using S2GeographyType = s2geog::PointGeography;

    MultiPoint(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

    inline GeographyType geog_type() const {
        return GeographyType::MultiPoint;
    }

    inline const std::vector<S2Point>& s2points() const {
        return static_cast<const s2geog::PointGeography&>(geog()).Points();
    }
};

class LineString : public Geography {
public:
    using S2GeographyType = s2geog::PolylineGeography;

    LineString(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

    inline GeographyType geog_type() const {
        return GeographyType::LineString;
    }

    inline const S2Polyline& s2polyline() const {
        const auto& polylines = static_cast<const S2GeographyType&>(geog()).Polylines();
        // TODO: does not work for empty point geography
        return *polylines[0];
    }
};

class MultiLineString : public Geography {
public:
    using S2GeographyType = s2geog::PolylineGeography;

    MultiLineString(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

    inline GeographyType geog_type() const {
        return GeographyType::MultiLineString;
    }

    inline const std::vector<std::unique_ptr<S2Polyline>>& s2polylines() const {
        return static_cast<const S2GeographyType&>(geog()).Polylines();
    }
};

class Polygon : public Geography {
public:
    using S2GeographyType = s2geog::PolygonGeography;

    Polygon(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

    inline GeographyType geog_type() const {
        return GeographyType::Polygon;
    }

    inline const S2Polygon& polygon() const {
        return *static_cast<const S2GeographyType&>(geog()).Polygon();
    }
};

class GeographyCollection : public Geography {
public:
    using S2GeographyType = s2geog::GeographyCollection;

    GeographyCollection(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)) {};

    inline GeographyType geog_type() const {
        return GeographyType::GeographyCollection;
    }

    const std::vector<std::unique_ptr<s2geog::Geography>>& features() const {
        return static_cast<const S2GeographyType&>(geog()).Features();
    }
};

/*
** Helper to create Geography object wrappers.
**
** @tparam T1 The S2Geography wrapper type
** @tparam T2 This library wrapper type.
** @tparam S The S2Geometry type
*/
template <class T1, class T2, class S>
std::unique_ptr<T2> make_geography(S&& s2_obj) {
    S2GeographyPtr s2geog_ptr = std::make_unique<T1>(std::forward<S>(s2_obj));
    return std::make_unique<T2>(std::move(s2geog_ptr));
}

// Helpers for explicit copy of s2geography objects.
std::unique_ptr<s2geog::Geography> clone_s2geography(const s2geog::Geography& geog);

}  // namespace spherely

#endif  // SPHERELY_GEOGRAPHY_H_
