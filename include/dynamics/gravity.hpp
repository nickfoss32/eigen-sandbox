#pragma once

#include <Eigen/Dense>

class IGravityModel {
public:
    virtual ~IGravityModel() = default;
    virtual Eigen::Vector3d compute_force(const Eigen::Vector3d& r) const = 0;
};

class PointMassGravity : public IGravityModel {
private:
    double GM_; // Gravitational constant * Earth mass (m^3/s^2)

public:
    PointMassGravity(double GM = 3.986004418e14) : GM_(GM) {}

    Eigen::Vector3d compute_force(const Eigen::Vector3d& r) const override {
        double r_norm = r.norm();
        return -GM_ / (r_norm * r_norm * r_norm) * r;
    }
};

class J2Gravity : public IGravityModel {
private:
    double GM_; // Gravitational constant * Earth mass (m^3/s^2)
    double J2_; // J2 coefficient
    double Re_; // Equatorial radius (m)

public:
    J2Gravity(double GM = 3.986004418e14, double J2 = 1.08262668e-3, double Re = 6378137.0)
        : GM_(GM), J2_(J2), Re_(Re) {}

    Eigen::Vector3d compute_force(const Eigen::Vector3d& r) const override {
        double r_norm = r.norm();
        double r2 = r_norm * r_norm;
        double r5 = r2 * r2 * r_norm;

        // Point-mass term
        Eigen::Vector3d a = -GM_ / (r_norm * r2) * r;

        // J2 perturbation
        double z_over_r = r(2) / r_norm;
        double z_over_r2 = z_over_r * z_over_r;
        Eigen::Vector3d a_j2;
        a_j2 << r(0) * (1.0 - 5.0 * z_over_r2),
                r(1) * (1.0 - 5.0 * z_over_r2),
                r(2) * (3.0 - 5.0 * z_over_r2);
        a_j2 *= 1.5 * GM_ * J2_ * Re_ * Re_ / r5;

        return a + a_j2;
    }
};
