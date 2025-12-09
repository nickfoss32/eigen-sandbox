#pragma once

#include <iostream>
#include <string>

namespace common {

/// @brief Enum to specify a coordinate frame
enum class CoordinateFrame {
    ECI,  // Earth-Centered Inertial
    ECEF  // Earth-Centered Earth-Fixed
};

std::istream& operator>>(std::istream& is, CoordinateFrame& frame);
std::ostream& operator<<(std::ostream& os, const CoordinateFrame& frame);

} // namespace common
