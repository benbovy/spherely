#include "s2/s2earth.h"

namespace spherely {

struct numeric_constants {
    static constexpr double EARTH_RADIUS_METERS = S2Earth::RadiusMeters();
};

}  // namespace spherely
