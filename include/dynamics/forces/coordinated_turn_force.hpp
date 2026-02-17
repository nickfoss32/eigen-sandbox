#pragma once

#include "dynamics/forces/force.hpp"

namespace dynamics {

/// @brief Coordinated turn force model in the XY plane.
///
/// Applies velocity-coupled acceleration:
/// ax = -omega * vy
/// ay =  omega * vx
/// az = 0
class CoordinatedTurnForce : public IForce {
public:
    /// @brief Constructor
    /// @param turn_rate_rad_s Turn rate in rad/s about +Z
    explicit CoordinatedTurnForce(double turn_rate_rad_s);

    /// @brief Compute coordinated-turn acceleration from velocity
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Compute Jacobians with respect to position and velocity
    auto compute_jacobian(const ForceContext& ctx) const
        -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    double omega_;
};

} // namespace dynamics
