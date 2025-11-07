#include "gravity.hpp"

namespace dynamics {
PointMassGravity::PointMassGravity(double GM)
: GM_(GM)
{}

Eigen::Vector3d PointMassGravity::compute_force(const Eigen::Vector3d& r) const {
    double r_norm = r.norm();
    return -GM_ / (r_norm * r_norm * r_norm) * r;
}

J2Gravity::J2Gravity(double GM, double J2, double Re)
: GM_(GM), J2_(J2), Re_(Re)
{}

Eigen::Vector3d J2Gravity::compute_force(const Eigen::Vector3d& r) const {
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
} // namespace dynamics
