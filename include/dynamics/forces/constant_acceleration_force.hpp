#pragma once

#include "dynamics/forces/force.hpp"

namespace dynamics {

/// @brief Constant acceleration force model.
///
/// Applies a fixed acceleration vector independent of position, velocity, and time:
/// a = a0
class ConstantAccelerationForce : public IForce {
public:
    /// @brief Constructor
    /// @param acceleration Constant acceleration in m/s^2
    explicit ConstantAccelerationForce(const Eigen::Vector3d& acceleration);

    /// @brief Compute constant acceleration
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Compute Jacobians (zero for constant acceleration)
    auto compute_jacobian(const ForceContext& ctx) const
        -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    Eigen::Vector3d acceleration_;
};

} // namespace dynamics
