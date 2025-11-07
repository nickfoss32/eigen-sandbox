#pragma once

#include <Eigen/Dense>

namespace dynamics {
class IGravityModel {
public:
    /// @brief Virtual destructor
    virtual ~IGravityModel() = default;

    /// @brief Interface for computing gravitational acceleration at position r
    /// @param r Position vector (m)
    virtual Eigen::Vector3d compute_force(const Eigen::Vector3d& r) const = 0;
};

class PointMassGravity : public IGravityModel {
private:
    double GM_; ///< Gravitational constant * Earth mass (m^3/s^2)

public:
    /// @brief Constructor
    /// @param GM Gravitational constant * Earth mass (default is Earth's value)
    PointMassGravity(double GM = 3.986004418e14);

    /// @brief Computes gravitational acceleration at position r
    /// @param r Position vector (m)
    Eigen::Vector3d compute_force(const Eigen::Vector3d& r) const override;
};

class J2Gravity : public IGravityModel {
private:
    double GM_; ///< Gravitational constant * Earth mass (m^3/s^2)
    double J2_; ///< J2 coefficient
    double Re_; ///< Equatorial radius (m)

public:
    /// @brief Constructor
    /// @param GM Gravitational constant * Earth mass (default is Earth's value)
    /// @param J2 J2 coefficient (default is Earth's value)
    /// @param Re Equatorial radius (default is Earth's value in meters)
    J2Gravity(double GM = 3.986004418e14, double J2 = 1.08262668e-3, double Re = 6378137.0);

    /// @brief Computes gravitational acceleration at position r including J2 perturbation
    Eigen::Vector3d compute_force(const Eigen::Vector3d& r) const override;
};
} // namespace dynamics
