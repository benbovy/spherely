#ifndef SPHERELY_PROJECTIONS_H_
#define SPHERELY_PROJECTIONS_H_

#include <s2/s2latlng.h>
#include <s2/s2projections.h>
#include <s2geography.h>

#include "pybind11.hpp"

namespace s2geog = s2geography;
using namespace spherely;

class Projection {
public:
    Projection(std::shared_ptr<S2::Projection> projection)
        : m_s2_projection(std::move(projection)) {}

    std::shared_ptr<S2::Projection> s2_projection() {
        return m_s2_projection;
    }

    static Projection lnglat() {
        return Projection(std::move(s2geog::lnglat()));
    }
    static Projection pseudo_mercator() {
        return Projection(std::move(s2geog::pseudo_mercator()));
    }
    static Projection orthographic(double longitude, double latitude) {
        return Projection(
            std::move(s2geog::orthographic(S2LatLng::FromDegrees(latitude, longitude))));
    }

private:
    std::shared_ptr<S2::Projection> m_s2_projection;
};

#endif  // SPHERELY_PROJECTIONS_H_
