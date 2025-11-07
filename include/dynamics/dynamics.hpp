#pragma once

#include <Eigen/Dense>
#include <stdexcept>

namespace dynamics {
/// @brief Enum to specify a coordinate frame
enum class CoordinateFrame {
    ECI,  // Earth-Centered Inertial
    ECEF  // Earth-Centered Earth-Fixed
};

inline std::istream& operator>>(std::istream& is, CoordinateFrame& frame) {
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

inline std::ostream& operator<<(std::ostream& os, const CoordinateFrame& frame) {
    switch (frame) {
        case CoordinateFrame::ECI: os << "ECI"; break;
        case CoordinateFrame::ECEF: os << "ECEF"; break;
    }
    return os;
}

/// @brief Base class defining how a system evolves by specifying the derivative of the state vector with respect to time
///        based on the physical laws governing the system.
class IDynamics {
public:
    /// @brief Constructor
    /// @param coordinateFrame Coordinate frame the dynamics object is configured to use.
    IDynamics(CoordinateFrame coordinateFrame)
     : coordinateFrame_(coordinateFrame)
    {}

    /// @brief virtual dtor
    virtual ~IDynamics() = default;
    
    /// @brief Computes the time derivative of the state vector at a given time.
    /// @param t Current time (in seconds).
    /// @param state Current state vector (e.g., position and velocity components).
    /// @return The derivative of the state vector (e.g., velocities and accelerations).
    virtual Eigen::VectorXd derivative(double t, const Eigen::VectorXd& state) const = 0;

    /// @brief The coordinate frame the dyanmics object is configured for.
    const CoordinateFrame coordinateFrame_{CoordinateFrame::ECEF};
};
} // namespace dynamics
