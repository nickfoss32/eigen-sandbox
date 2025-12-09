#pragma once

#include "dynamics/force.hpp"

#include <Eigen/Dense>

#include <utility>

namespace dynamics {

/// @brief Point mass gravity model
/// @details Implements Newtonian gravity: a = -μ/r³ * r
class PointMassGravity : public IForce {
public:
    /// @brief Constructor
    /// @param GM Gravitational constant * mass (default is Earth's value)
    explicit PointMassGravity(double GM = 3.986004418e14);

    /// @brief Computes gravitational acceleration at position
    /// @details computes: f(x,v) = a = -μ/r³ * r
    auto compute_force(const ForceContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Computes Jacobian of gravitational acceleration
    /// @details computes: da/d
    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    double GM_; ///< Gravitational parameter (m^3/s^2)
};

/// @brief J2 gravity perturbation model
class J2Gravity : public IForce {
public:
    /// @brief Constructor
    /// @param GM Gravitational constant * mass (default is Earth's value)
    /// @param J2 J2 coefficient (default is Earth's value)
    /// @param Re Equatorial radius (default is Earth's value in meters)
    explicit J2Gravity(double GM = 3.986004418e14, double J2 = 1.08262668e-3, double Re = 6378137.0);

    /// @brief Computes gravitational acceleration including J2 perturbation
    auto compute_force(const ForceContext& ctx) const -> Eigen::Vector3d override;

    /// @brief Computes Jacobian of gravitational acceleration
    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override;

private:
    double GM_; ///< Gravitational parameter (m^3/s^2)
    double J2_; ///< J2 coefficient
    double Re_; ///< Equatorial radius (m)
};

} // namespace dynamics
