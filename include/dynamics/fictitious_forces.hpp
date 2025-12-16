#pragma once

#include "dynamics/force.hpp"

#include <Eigen/Dense>

namespace dynamics {

/// @brief Fictitious forces for rotating reference frames (Coriolis and centrifugal)
class FictitiousForces : public IForce {
public:
    /// @brief Constructor
    /// @param omega Angular velocity vector (rad/s) - default is Earth's rotation
    explicit FictitiousForces(const Eigen::Vector3d& omega = Eigen::Vector3d(0.0, 0.0, 7.292115e-5));

    /// @brief Computes fictitious accelerations (Coriolis + centrifugal)
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Compute force Jacobian
    /// @param t Time
    /// @param state State vector [position, velocity]
    /// @return pair of (∂a/∂r, ∂a/∂v)
    virtual auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    Eigen::Vector3d omega_; ///< Angular velocity vector (rad/s)
};

} // namespace dynamics
