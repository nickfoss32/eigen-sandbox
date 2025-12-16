#include "dynamics/atmospheric_drag.hpp"
#include <cmath>

namespace dynamics {

auto AtmosphericDrag::compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d {
    // Get velocity magnitude
    const double v_mag = ctx.velocity.norm();
    
    // No drag if stationary
    if (v_mag < 1e-10) {
        return Eigen::Vector3d::Zero();
    }
    
    // Compute altitude above Earth surface
    const double r_mag = ctx.position.norm();
    const double altitude = r_mag - R_earth_;
    
    // Get atmospheric density at current altitude
    const double rho = atmosphere::get_density_us76(altitude);
    
    // Drag acceleration: a = -0.5 * ρ * v² * (C_d * A / m) * v̂
    const double drag_coeff = 0.5 * rho * v_mag * (Cd_ * A_ / m_);
    const Eigen::Vector3d velocity_unit = ctx.velocity / v_mag;
    
    return -drag_coeff * velocity_unit * v_mag;
}

auto AtmosphericDrag::compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {
    const double v_mag = ctx.velocity.norm();
    
    // If velocity is near zero, both Jacobians are zero
    if (v_mag < 1e-10) {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    
    // --- Compute ∂a/∂r (density variation with position) ---
    
    const double r_mag = ctx.position.norm();
    const double altitude = r_mag - R_earth_;
    const double rho = atmosphere::get_density_us76(altitude);
    
    // Numerical derivative of density with respect to altitude
    constexpr double h_step = 1.0;  // 1 meter step
    const double rho_plus = atmosphere::get_density_us76(altitude + h_step);
    const double rho_minus = atmosphere::get_density_us76(altitude - h_step);
    const double drho_dh = (rho_plus - rho_minus) / (2.0 * h_step);
    
    // Altitude derivative: ∂h/∂r = ∂(|r| - R_earth)/∂r = r̂
    const Eigen::Vector3d r_unit = ctx.position / r_mag;
    
    // Drag acceleration (without sign): a_mag = 0.5 * ρ * v² * (C_d * A / m)
    const double base_coeff = 0.5 * v_mag * v_mag * (Cd_ * A_ / m_);
    const Eigen::Vector3d velocity_unit = ctx.velocity / v_mag;
    
    // ∂a/∂r = -∂(ρ * base_coeff * v̂)/∂r = -base_coeff * v̂ * (drho_dh * r̂ᵀ)
    Eigen::Matrix3d da_dr = -base_coeff * drho_dh * velocity_unit * r_unit.transpose();
    
    // --- Compute ∂a/∂v (velocity dependence) ---
    
    // Drag force: F = -0.5 * ρ * |v| * v * (C_d * A / m)
    //           a = -0.5 * ρ * (C_d * A / m) * |v| * v
    //
    // Let k = -0.5 * ρ * (C_d * A / m)
    // Then a = k * |v| * v
    //
    // ∂a/∂v = k * ∂(|v| * v)/∂v
    //       = k * (|v| * I + v * ∂|v|/∂v)
    //       = k * (|v| * I + v * (v/|v|)ᵀ)
    //       = k * (|v| * I + v * v̂ᵀ)
    
    const double k = -0.5 * rho * (Cd_ * A_ / m_);
    Eigen::Matrix3d da_dv = k * (v_mag * Eigen::Matrix3d::Identity() + ctx.velocity * velocity_unit.transpose());
    
    return {da_dr, da_dv};
}

} // namespace dynamics
