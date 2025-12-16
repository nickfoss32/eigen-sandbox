#pragma once

#include "force.hpp"
#include "atmosphere.hpp"
#include <Eigen/Dense>

namespace dynamics {

/// @brief Atmospheric drag force model
/// 
/// @details Computes drag force opposing the velocity vector:
///          
///          F_drag = -0.5 * ρ(h) * |v|² * C_d * A * v̂
///          
///          where:
///          - ρ(h) is atmospheric density at altitude h (kg/m³)
///          - v is velocity vector (m/s)
///          - C_d is drag coefficient (dimensionless)
///          - A is reference cross-sectional area (m²)
///          - v̂ = v/|v| is the velocity unit vector
///          
///          The acceleration is:
///          a_drag = F_drag / m = -0.5 * ρ(h) * |v|² * (C_d * A / m) * v̂
///          
///          This force:
///          - Always opposes motion (negative sign)
///          - Scales with velocity squared
///          - Depends on altitude through density ρ(h)
///          - Is velocity-dependent but time-independent (autonomous)
///          
/// @note The ballistic coefficient β = m / (C_d * A) determines drag sensitivity.
///       Larger β means less drag effect (e.g., dense, streamlined objects).
class AtmosphericDrag : public IForce {
public:
    /// @brief Constructs an atmospheric drag model
    /// @param mass Object mass (kg)
    /// @param drag_coefficient Drag coefficient C_d (dimensionless, typically 1.0-2.5)
    /// @param reference_area Reference cross-sectional area (m²)
    /// @param earth_radius Radius of Earth for altitude calculation (m), default = 6378137 m
    AtmosphericDrag(double mass, double drag_coefficient, double reference_area, 
                    double earth_radius = 6378137.0)
        : m_(mass)
        , Cd_(drag_coefficient)
        , A_(reference_area)
        , R_earth_(earth_radius)
    {}
    
    /// @brief Computes drag acceleration a = -0.5 * ρ * v² * (C_d * A / m) * v̂
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override;
    
    /// @brief Computes Jacobians ∂a/∂r and ∂a/∂v
    /// 
    /// @details The drag force depends on both position (through altitude/density)
    ///          and velocity (quadratic dependence). The Jacobians are:
    ///          
    ///          ∂a/∂r: Captures density variation with altitude
    ///          ∂a/∂v: Captures velocity-dependent drag magnitude and direction
    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override;
    
private:
    double m_;        ///< Mass (kg)
    double Cd_;       ///< Drag coefficient (dimensionless)
    double A_;        ///< Reference area (m²)
    double R_earth_;  ///< Earth radius for altitude calculation (m)
};

} // namespace dynamics
