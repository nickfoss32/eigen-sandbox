#pragma once

#include "dynamics/forces/force.hpp"

#include <memory>

namespace dynamics {

/// @brief Activates an underlying force only within a time window.
///
/// This is useful for truth scenarios with scheduled maneuvers or boost phases
/// while still staying inside the point-mass force composition framework.
class TimeWindowForce : public IForce {
public:
    /// @brief Constructor.
    /// @param inner_force Force to apply while the window is active.
    /// @param start_time Start of the active window in seconds.
    /// @param end_time End of the active window in seconds.
    TimeWindowForce(
        std::shared_ptr<IForce> inner_force,
        double start_time,
        double end_time
    );

    /// @brief Compute acceleration from the wrapped force when active.
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Compute wrapped Jacobians when active, otherwise zeros.
    auto compute_jacobian(const ForceContext& ctx) const
        -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    /// @brief Check whether the current time is inside the active window.
    auto is_active(double time_seconds) const -> bool;

    std::shared_ptr<IForce> inner_force_;
    double start_time_;
    double end_time_;
};

} // namespace dynamics
