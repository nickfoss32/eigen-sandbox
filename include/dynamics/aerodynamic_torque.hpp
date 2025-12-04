#pragma once

#include "dynamics/torque.hpp"

namespace dynamics {

/// @brief Aerodynamic torque model
///
/// Simple model based on offset between center of pressure and center of mass.
class AerodynamicTorque : public ITorque {
public:
    /// @param center_of_pressure Center of pressure in body frame (m)
    /// @param center_of_mass Center of mass in body frame (m)
    /// @param drag_coefficient Drag coefficient
    /// @param reference_area Reference area (m²)
    AerodynamicTorque(
        const Eigen::Vector3d& center_of_pressure,
        const Eigen::Vector3d& center_of_mass,
        double drag_coefficient,
        double reference_area
    );

    /// @brief Computes the aerodynamic torque τ = f(q, ω, r, v, t)
    auto compute_torque(const TorqueContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Computes the Jacobian matrices ∂τ/∂q, ∂τ/∂ω, ∂τ/∂r, ∂τ/∂v
    auto compute_jacobian(const TorqueContext& ctx) const -> std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    Eigen::Vector3d cop_;     ///< Center of pressure (body frame)
    Eigen::Vector3d com_;     ///< Center of mass (body frame)
    double cd_;               ///< Drag coefficient
    double area_;             ///< Reference area (m²)
};

} // namespace dynamics
