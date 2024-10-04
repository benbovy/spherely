#include "s2geography_addons.hpp"

#include <memory>
#include <sstream>
#include <string>

#include "s2/s2cap.h"
#include "s2/s2cell.h"
#include "s2/s2edge_crosser.h"
#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"
#include "s2/s2latlng_rect_bounder.h"
#include "s2/s2lax_loop_shape.h"
#include "s2/s2region.h"
#include "s2/s2shape.h"

using namespace s2geography;

// This class is a shim to allow a class to return a std::unique_ptr<S2Shape>(),
// which is required by MutableS2ShapeIndex::Add(), without copying the
// underlying data. S2Shape instances do not typically own their data (e.g.,
// S2Polygon::Shape), so this does not change the general relationship (that
// anything returned by Geography::Shape() is only valid within the scope of
// the Geography). Note that this class is also available (but not exposed) in
// s2/s2shapeutil_coding.cc and s2geography/geography.cc.
class S2ShapeWrapper : public S2Shape {
public:
    S2ShapeWrapper(S2Shape* shape) : m_shape(shape) {}
    int num_edges() const {
        return m_shape->num_edges();
    }
    Edge edge(int edge_id) const {
        return m_shape->edge(edge_id);
    }
    int dimension() const {
        return m_shape->dimension();
    }
    ReferencePoint GetReferencePoint() const {
        return m_shape->GetReferencePoint();
    }
    int num_chains() const {
        return m_shape->num_chains();
    }
    Chain chain(int chain_id) const {
        return m_shape->chain(chain_id);
    }
    Edge chain_edge(int chain_id, int offset) const {
        return m_shape->chain_edge(chain_id, offset);
    }
    ChainPosition chain_position(int edge_id) const {
        return m_shape->chain_position(edge_id);
    }

private:
    S2Shape* m_shape;
};

// Implements the S2Region interface for a S2LaxClosedPolylineShape object
// (copied mostly from the S2Polyline implementation)
class S2ClosedPolylineRegion : public S2Region {
public:
    S2ClosedPolylineRegion(S2LaxClosedPolylineShape* shape_ptr) : m_shape_ptr(shape_ptr) {}
    ~S2ClosedPolylineRegion() = default;

    S2ClosedPolylineRegion* Clone() const override {
        return new S2ClosedPolylineRegion(*this);
    }

    S2Cap GetCapBound() const override {
        return GetRectBound().GetCapBound();
    }
    S2LatLngRect GetRectBound() const override {
        S2LatLngRectBounder bounder;
        for (int i = 0; i < m_shape_ptr->num_vertices(); ++i) {
            bounder.AddPoint(m_shape_ptr->vertex(i));
        }
        return bounder.GetBound();
    }

    bool Contains(const S2Cell& /*cell*/) const override {
        return false;
    }
    bool Contains(const S2Point& /*p*/) const override {
        return false;
    }

    bool MayIntersect(const S2Cell& cell) const override {
        if (m_shape_ptr->num_vertices() == 0) return false;

        for (int i = 0; i < m_shape_ptr->num_vertices(); ++i) {
            if (cell.Contains(m_shape_ptr->vertex(i))) return true;
        }
        S2Point cell_vertices[4];
        for (int i = 0; i < 4; ++i) {
            cell_vertices[i] = cell.GetVertex(i);
        }
        for (int j = 0; j < 4; ++j) {
            S2EdgeCrosser crosser(
                &cell_vertices[j], &cell_vertices[(j + 1) & 3], &m_shape_ptr->vertex(0));
            for (int i = 1; i < m_shape_ptr->num_vertices(); ++i) {
                if (crosser.CrossingSign(&m_shape_ptr->vertex(i)) >= 0) {
                    return true;
                }
            }
        }
        return false;
    }

private:
    S2LaxClosedPolylineShape* m_shape_ptr;
};

std::unique_ptr<S2Shape> ClosedPolylineGeography::Shape(int /*id*/) const {
    return std::make_unique<S2ShapeWrapper>(m_polyline_ptr.get());
}

std::unique_ptr<S2Region> ClosedPolylineGeography::Region() const {
    return std::make_unique<S2ClosedPolylineRegion>(m_polyline_ptr.get());
}

void ClosedPolylineGeography::GetCellUnionBound(std::vector<S2CellId>* cell_ids) const {
    return S2ClosedPolylineRegion(m_polyline_ptr.get()).GetCellUnionBound(cell_ids);
}

const std::string ClosedPolylineGeography::wkt(int significant_digits) const {
    std::stringstream wkt_stream;
    char write_buffer[1024];

    auto fmt_coord = [&](double value) {
        auto length = snprintf(write_buffer, 128, "%.*g", significant_digits, value);
        return std::string(write_buffer, static_cast<std::size_t>(length));
    };

    wkt_stream << "LINEARRING (";

    for (int i = 0; i < m_polyline_ptr->num_vertices(); i++) {
        S2LatLng ll(m_polyline_ptr->vertex(i));

        wkt_stream << fmt_coord(ll.lng().degrees()) << " " << fmt_coord(ll.lat().degrees());
        wkt_stream << ", ";
    }

    // close ring (add 1st coordinate pair)
    S2LatLng ll0(m_polyline_ptr->vertex(0));
    wkt_stream << fmt_coord(ll0.lng().degrees()) << " " << fmt_coord(ll0.lat().degrees());

    wkt_stream << ")";

    return wkt_stream.str();
}
