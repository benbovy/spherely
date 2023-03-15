#ifndef S2GEOGRAPHY_ADDONS_H_
#define S2GEOGRAPHY_ADDONS_H_

// Things that are not (yet) implemented in s2geography and that might
// eventually be moved there.

#include <memory>
#include <string>
#include <vector>

#include "s2/s2lax_loop_shape.h"
#include "s2/s2loop.h"
#include "s2geography/geography.h"

namespace s2geography {

/*
** A Geography representing a single closed polyline.
**
** Unlike Point, Polyline and Polygon, s2geometry has no S2Region subclass for
** representing a closed polyline with no interior, so this wrapper class uses
** the S2LaxClosedPolylineShape class as the underlying representation.
**
** It can be built from a (temporary) S2Loop object for validating the geometry
** (note: the CCW vs. CW vertices order has little importance here but automated
** closing and other checks like number of vertices, duplicates vertices and
** interesction are still relevant).
*/
class ClosedPolylineGeography : public Geography {
public:
    ClosedPolylineGeography() {}
    ClosedPolylineGeography(const S2Loop& loop)
        : m_polyline_ptr(new S2LaxClosedPolylineShape(loop)) {}

    int dimension() const {
        return 1;
    }
    int num_shapes() const {
        return 1;
    }

    std::unique_ptr<S2Shape> Shape(int /*id*/) const;
    std::unique_ptr<S2Region> Region() const;
    void GetCellUnionBound(std::vector<S2CellId>* cell_ids) const;

    // This should eventually be moved into s2geography::WKTWriter
    // (or we need to refactor s2geography::WKTWriter so that we can extend it)
    const std::string wkt(int significant_digits = 16) const;

private:
    std::unique_ptr<S2LaxClosedPolylineShape> m_polyline_ptr;
};

}  // namespace s2geography

#endif  // S2GEOGRAPHY_ADDONS_H_
