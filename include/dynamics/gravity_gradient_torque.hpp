#pragma once

#include "dynamics/torque.hpp"

namespace dynamics {

/// @brief Gravity gradient torque for attitude dynamics
///
/// Torque due to differential gravitational force across the body.
/// Important for satellites and objects in orbit.
class GravityGradientTorque : public ITorque {
public:
    /// @brief Constructor
    /// @param inertia Inertia tensor in body frame (kg·m²)
    /// @param GM Gravitational parameter (m³/s²)
    GravityGradientTorque(const Eigen::Matrix3d& inertia, double GM = 3.986004418e14);

    /// @brief Computes the gravity gradient torque τ = f(q, ω, r, v, t)
    /// @param ctx Torque context containing time, orbital state, and attitude state
    /// @return Torque vector in body frame (N·m)
    auto compute_torque(const TorqueContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Computes the Jacobian matrices ∂τ/∂q, ∂τ/∂ω, ∂τ/∂r, ∂τ/∂v
    /// @param ctx Torque context containing time, orbital state, and attitude state
    /// @return Tuple of Jacobian matrices (∂τ/∂q, ∂τ/∂ω, ∂τ/∂r, ∂τ/∂v)
    auto compute_jacobian(const TorqueContext& ctx) const -> std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    Eigen::Matrix3d inertia_; ///< Inertia tensor (kg·m²)
    double GM_; ///< Gravitational parameter (m³/s²)
};

} // namespace dynamics
