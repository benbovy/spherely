#ifndef SPHERELY_GEOGRAPHY_H_
#define SPHERELY_GEOGRAPHY_H_

#include <memory>

#include "s2/s2loop.h"
#include "s2/s2point.h"
#include "s2/s2polyline.h"
#include "s2geography.h"
#include "s2geography_addons.hpp"

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
    LinearRing,
    Polygon,
    MultiPoint,
    MultiLineString
};

/*
** Thin wrapper around s2geography::Geography.
**
** Implements move semantics (avoid implicit copies).
** Allows getting the geography type.
** Wraps a s2geography::ShapeIndexGeography to speed-up operations
** (the index is built on demand).
**
*/
class Geography {
public:
    using S2GeographyType = s2geog::Geography;

    Geography(const Geography&) = delete;
    Geography(Geography&& geog) : m_s2geog_ptr(std::move(geog.m_s2geog_ptr)) {
        // std::cout << "Geography move constructor called: " << this <<
        // std::endl;
    }
    Geography(S2GeographyPtr&& s2geog_ptr) : m_s2geog_ptr(std::move(s2geog_ptr)) {}

    ~Geography() {
        // std::cout << "Geography destructor called: " << this << std::endl;
    }

    Geography& operator=(const Geography&) = delete;
    Geography& operator=(Geography&& other) {
        // std::cout << "Geography move assignment called: " << this <<
        // std::endl;
        m_s2geog_ptr = std::move(other.m_s2geog_ptr);
        return *this;
    }

    inline virtual GeographyType geog_type() const {
        return GeographyType::None;
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

private:
    S2GeographyPtr m_s2geog_ptr;
    S2GeographyIndexPtr m_s2geog_index_ptr;
};

class Point : public Geography {
public:
    using S2GeographyType = s2geog::PointGeography;

    Point(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
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

    MultiPoint(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::MultiPoint;
    }
};

class LineString : public Geography {
public:
    using S2GeographyType = s2geog::PolylineGeography;

    LineString(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::LineString;
    }

    inline const S2Polyline& s2polyline() const {
        const auto& polylines = static_cast<const s2geog::PolylineGeography&>(geog()).Polylines();
        // TODO: does not work for empty point geography
        return *polylines[0];
    }
};

class MultiLineString : public Geography {
public:
    using S2GeographyType = s2geog::PolylineGeography;

    MultiLineString(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::MultiLineString;
    }
};

class LinearRing : public Geography {
public:
    using S2GeographyType = s2geog::ClosedPolylineGeography;

    LinearRing(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::LinearRing;
    }
};

class Polygon : public Geography {
public:
    using S2GeographyType = s2geog::PolygonGeography;

    Polygon(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::Polygon;
    }
};

}  // namespace spherely

#endif  // SPHERELY_GEOGRAPHY_H_
