#ifndef SPHERELY_GEOGRAPHY_H_
#define SPHERELY_GEOGRAPHY_H_

#include <pybind11/pybind11.h>
#include <s2geography/geography.h>

#include <exception>
#include <memory>
#include <string>

namespace py = pybind11;
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
    Polygon = 3,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection
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
    Geography(Geography&& geog)
        : m_s2geog_ptr(std::move(geog.m_s2geog_ptr)),
          m_is_empty(geog.is_empty()),
          m_geog_type(geog.geog_type()) {}

    Geography(S2GeographyPtr&& s2geog_ptr) : m_s2geog_ptr(std::move(s2geog_ptr)) {
        // TODO: template constructors with s2geography Geography subclass constraints (e.g., using
        // SFINAE or "if constexpr") may be more efficient than dynamic casting like done in
        // extract_geog_properties.
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

    template <class T>
    inline const T* cast_geog() const noexcept {
        return reinterpret_cast<const T*>(&geog());
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
    std::unique_ptr<s2geog::Geography> clone_geog() const;

    py::tuple encode() const;
    static Geography decode(const py::tuple& encoded);

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

/**
 * Custom exception that may be thrown when an empty Geography is found.
 */
class EmptyGeographyException : public std::exception {
private:
    std::string message;

public:
    EmptyGeographyException(const char* msg) : message(msg) {}

    const char* what() const throw() {
        return message.c_str();
    }
};

// TODO: cleaner way? Already implemented elsewhere?
inline std::string format_geog_type(GeographyType geog_type) {
    if (geog_type == GeographyType::Point) {
        return "POINT";
    } else if (geog_type == GeographyType::MultiPoint) {
        return "MULTIPOINT";
    } else if (geog_type == GeographyType::LineString) {
        return "LINESTRING";
    } else if (geog_type == GeographyType::MultiLineString) {
        return "MULTILINESTRING";
    } else if (geog_type == GeographyType::Polygon) {
        return "POLYGON";
    } else if (geog_type == GeographyType::MultiPolygon) {
        return "MULTIPOLYGON";
    } else if (geog_type == GeographyType::GeometryCollection) {
        return "GEOMETRYCOLLECTION";
    } else {
        return "UNKNOWN";
    }
}

/**
 * Check the type of a Geography object and maybe raise an exception.
 */
inline void check_geog_type(const Geography& geog_obj, GeographyType geog_type) {
    if (geog_obj.geog_type() != geog_type) {
        auto expected = format_geog_type(geog_type);
        auto actual = format_geog_type(geog_obj.geog_type());

        throw py::type_error("invalid Geography type (expected " + expected + ", found " + actual +
                             ")");
    }
}

}  // namespace spherely

#endif  // SPHERELY_GEOGRAPHY_H_
