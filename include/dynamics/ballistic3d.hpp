#pragma once

#include <memory>

#include <Eigen/Dense>

#include "dynamics.hpp"
#include "gravity.hpp"

namespace dynamics {
/// @brief Dynamics model for a 3D ballistic trajectory under constant gravity with no other forces.
class Ballistic3D : public IDynamics {
public:
    /// @brief Constructs a 3D ballistic dynamics model with specified gravity.
    /// @param coordinateFrame Coordinate frame the dynamics object is configured to use.
    /// @param gravityModel gravity model to use for this system.
    Ballistic3D(CoordinateFrame coordinateFrame, std::shared_ptr<IGravityModel> gravityModel);

    /// @brief Computes the time derivative of the state vector for a 3D ballistic trajectory.
    /// @param t Current time (in seconds).
    /// @param state 6D state vector [x, y, z, vx, vy, vz] representing position and velocity.
    /// @return 6D derivative vector [dx/dt, dy/dt, dz/dt, d2x/dt, d2y/dt, d2z/dt] representing velocities and accelerations.
    Eigen::VectorXd derivative(double t, const Eigen::VectorXd& state) const override;

private:
    /// @brief gravity model for this system's dynamics
    std::shared_ptr<IGravityModel> gravityModel_;
    // std::shared_ptr<IThrustModel> thrustModel_;
    // std::shared_ptr<IDragModel> dragModel_;
};
} // namespace dynamics
