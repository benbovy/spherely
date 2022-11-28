#ifndef S2SHAPELY_GEOGRAPHY_H_
#define S2SHAPELY_GEOGRAPHY_H_

#include "s2geography.h"

namespace s2geog = s2geography;

namespace s2shapely {

using S2GeographyPtr = std::unique_ptr<s2geog::Geography>;
using S2GeographyIndexPtr = std::unique_ptr<s2geog::ShapeIndexGeography>;

/*
** The registered Geography types
*/
enum class GeographyType : std::int8_t { None = -1, Point, LineString };

/*
** Thin wrapper around s2geography::Geography.
**
** Implements move semantics (avoid implicit copies).
** Provides a geom_type static member.
** Wraps a s2geography::ShapeIndexGeography to speed-up operations
** built on demand only).
**
*/
class Geography {
   public:
    Geography(const Geography&) = delete;
    Geography(Geography&& geog) : m_s2geog_ptr(std::move(geog.m_s2geog_ptr)) {
        // std::cout << "Geography move constructor called: " << this <<
        // std::endl;
    }
    Geography(S2GeographyPtr&& s2geog_ptr)
        : m_s2geog_ptr(std::move(s2geog_ptr)) {}

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

    inline const virtual GeographyType geog_type() const {
        return GeographyType::None;
    }

    inline const s2geog::Geography& geog() const { return *m_s2geog_ptr; }

    inline const s2geog::ShapeIndexGeography& geog_index() {
        if (!m_s2geog_index_ptr) {
            m_s2geog_index_ptr =
                std::make_unique<s2geog::ShapeIndexGeography>(geog());
        }

        return *m_s2geog_index_ptr;
    }
    void reset_index() { m_s2geog_index_ptr.reset(); }
    bool has_index() { return m_s2geog_index_ptr != nullptr; }

    int dimension() const { return m_s2geog_ptr->dimension(); }
    int num_shapes() const { return m_s2geog_ptr->num_shapes(); }

   private:
    S2GeographyPtr m_s2geog_ptr;
    S2GeographyIndexPtr m_s2geog_index_ptr;
};

class Point : public Geography {
   public:
    Point(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline const GeographyType geog_type() const override {
        return GeographyType::Point;
    }
};

class LineString : public Geography {
   public:
    LineString(S2GeographyPtr&& geog_ptr) : Geography(std::move(geog_ptr)){};

    inline const GeographyType geog_type() const override {
        return GeographyType::LineString;
    }
};

}  // namespace s2shapely

#endif  // S2SHAPELY_GEOGRAPHY_H_
