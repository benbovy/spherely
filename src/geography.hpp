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
** - Add ``geog_type()`` method for getting the geography type
** - Eagerly infer the geography type as well as other properties
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

    template <class T>
    inline const T* downcast_geog() const {
        return dynamic_cast<const T*>(&geog());
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

}  // namespace spherely

#endif  // SPHERELY_GEOGRAPHY_H_
