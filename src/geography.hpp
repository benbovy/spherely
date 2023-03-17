#ifndef SPHERELY_GEOGRAPHY_H_
#define SPHERELY_GEOGRAPHY_H_

#include <memory>

#include "s2/s2lax_polyline_shape.h"
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
    MultiLineString,
    GeographyCollection
};

/*
** Thin wrapper around s2geography::Geography.
**
** This wrapper implements the following specific features (that might
** eventually move into s2geography::Geography?):
**
** - Implement move semantics only.
** - Add a virtual ``clone()`` method for explicit copies (similarly to s2geometry).
** - Add a virtual ``geog_type()`` method for getting the geography type.
** - Encapsulate a lazy ``s2geography::ShapeIndexGeography`` accessible via ``geog_index()``.
**
*/
class Geography {
public:
    using S2GeographyType = s2geog::Geography;

    Geography(const Geography&) = delete;
    Geography(Geography&& geog) : m_s2geog_ptr(std::move(geog.m_s2geog_ptr)) {}
    Geography(S2GeographyPtr&& s2geog_ptr) : m_s2geog_ptr(std::move(s2geog_ptr)) {}

    virtual ~Geography() {}

    Geography& operator=(const Geography&) = delete;
    Geography& operator=(Geography&& other) {
        m_s2geog_ptr = std::move(other.m_s2geog_ptr);
        return *this;
    }

    virtual Geography* clone() const = 0;

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

template <class T, std::enable_if_t<std::is_same_v<T, s2geog::PointGeography>, bool> = true>
std::unique_ptr<s2geog::Geography> clone_s2geography(const s2geog::Geography& geog) {
    const auto& points = static_cast<const T&>(geog).Points();
    return std::make_unique<T>(points);
}

template <class T, std::enable_if_t<std::is_same_v<T, s2geog::PolylineGeography>, bool> = true>
std::unique_ptr<s2geog::Geography> clone_s2geography(const s2geog::Geography& geog) {
    const auto& polylines = static_cast<const T&>(geog).Polylines();
    std::vector<std::unique_ptr<S2Polyline>> polylines_copy(polylines.size());

    auto copy_polyline = [](const std::unique_ptr<S2Polyline>& polyline) {
        return std::unique_ptr<S2Polyline>(polyline->Clone());
    };

    std::transform(polylines.begin(), polylines.end(), polylines_copy.begin(), copy_polyline);

    return std::make_unique<T>(std::move(polylines_copy));
}

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

    Point* clone() const override {
        return new Point(clone_s2geography<S2GeographyType>(geog()));
    }
};

class MultiPoint : public Geography {
public:
    using S2GeographyType = s2geog::PointGeography;

    MultiPoint(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::MultiPoint;
    }

    inline const std::vector<S2Point>& s2points() const {
        return static_cast<const s2geog::PointGeography&>(geog()).Points();
    }

    MultiPoint* clone() const override {
        return new MultiPoint(std::make_unique<s2geog::PointGeography>(s2points()));
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
        const auto& polylines = static_cast<const S2GeographyType&>(geog()).Polylines();
        // TODO: does not work for empty point geography
        return *polylines[0];
    }

    LineString* clone() const override {
        std::unique_ptr<S2Polyline> line_ptr(s2polyline().Clone());
        return new LineString(std::make_unique<S2GeographyType>(std::move(line_ptr)));
    }
};

class MultiLineString : public Geography {
public:
    using S2GeographyType = s2geog::PolylineGeography;

    MultiLineString(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::MultiLineString;
    }

    inline const std::vector<std::unique_ptr<S2Polyline>>& s2polylines() const {
        return static_cast<const S2GeographyType&>(geog()).Polylines();
    }

    MultiLineString* clone() const override {
        const auto& polylines = s2polylines();
        std::vector<std::unique_ptr<S2Polyline>> polylines_copy(polylines.size());

        auto copy_polyline = [](const std::unique_ptr<S2Polyline>& polyline) {
            return std::unique_ptr<S2Polyline>(polyline->Clone());
        };

        std::transform(polylines.begin(), polylines.end(), polylines_copy.begin(), copy_polyline);

        return new MultiLineString(std::make_unique<S2GeographyType>(std::move(polylines_copy)));
    }
};

class LinearRing : public Geography {
public:
    using S2GeographyType = s2geog::ClosedPolylineGeography;

    LinearRing(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::LinearRing;
    }

    inline const S2LaxClosedPolylineShape& s2polyline() const {
        return static_cast<const S2GeographyType&>(geog()).Polyline();
    }

    LinearRing* clone() const override {
        const auto& s2geog = static_cast<const S2GeographyType&>(geog());
        return new LinearRing(std::unique_ptr<S2GeographyType>(s2geog.clone()));
    }
};

class Polygon : public Geography {
public:
    using S2GeographyType = s2geog::PolygonGeography;

    Polygon(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::Polygon;
    }

    inline const S2Polygon& polygon() const {
        return *static_cast<const S2GeographyType&>(geog()).Polygon();
    }

    Polygon* clone() const override {
        std::unique_ptr<S2Polygon> poly_ptr(polygon().Clone());
        return new Polygon(std::make_unique<S2GeographyType>(std::move(poly_ptr)));
    }
};

class GeographyCollection : public Geography {
public:
    using S2GeographyType = s2geog::GeographyCollection;

    GeographyCollection(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline GeographyType geog_type() const override {
        return GeographyType::GeographyCollection;
    }

    const std::vector<std::unique_ptr<s2geog::Geography>>& features() const {
        return static_cast<const S2GeographyType&>(geog()).Features();
    }

    GeographyCollection* clone() const override {
        const auto& s2geog_features = features();
        std::vector<std::unique_ptr<s2geog::Geography>> features_copy;
        features_copy.reserve(s2geog_features.size());

        // Wrap each underlying s2geography::Geography feature into
        // spherely::Geography, call the clone method and get back the
        // wrapped s2geography::Geography object.
        //
        // This is an ugly hack that would be much nicer if
        // s2geography::Geography provided a virtual ``clone`` method
        for (const auto& feature_ptr : s2geog_features) {
            std::unique_ptr<s2geog::Geography> cloned;

            auto point_ptr = dynamic_cast<s2geog::PointGeography*>(feature_ptr.get());
            if (point_ptr != nullptr) {
                cloned = clone_s2geography<s2geog::PointGeography>(*point_ptr);
            } else {
                auto polyline_ptr = dynamic_cast<s2geog::PolylineGeography*>(feature_ptr.get());
                if (polyline_ptr != nullptr) {
                    cloned = clone_s2geography<s2geog::PolylineGeography>(*polyline_ptr);
                }
            }

            features_copy.push_back(std::move(cloned));
        }

        return new GeographyCollection(std::make_unique<S2GeographyType>(std::move(features_copy)));
    }
};

}  // namespace spherely

#endif  // SPHERELY_GEOGRAPHY_H_
