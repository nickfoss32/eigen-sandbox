#include "common/coordinate_frame.hpp"

#include <stdexcept>

namespace common {

std::istream& operator>>(std::istream& is, CoordinateFrame& frame) {
    std::string s;
    is >> s;
    if (s == "ECI" || s == "eci") {
        frame = CoordinateFrame::ECI;
    } else if (s == "ECEF" || s == "ecef") {
        frame = CoordinateFrame::ECEF;
    } else {
        throw std::invalid_argument("Invalid coordinate frame: " + s);
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, const CoordinateFrame& frame) {
    switch (frame) {
        case CoordinateFrame::ECI: os << "ECI"; break;
        case CoordinateFrame::ECEF: os << "ECEF"; break;
    }
    return os;
}

} // namespace common
