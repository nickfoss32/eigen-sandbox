#include "dynamics/gravity.hpp"

namespace dynamics {

PointMassGravity::PointMassGravity(double GM)
    : GM_(GM)
{}

auto PointMassGravity::compute_force(const ForceContext& ctx) const -> Eigen::Vector3d {
    double r_norm = ctx.position.norm();
    return -GM_ / (r_norm * r_norm * r_norm) * ctx.position;
}

auto PointMassGravity::compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {
    Eigen::Vector3d r = ctx.position;
    double r_norm = r.norm();
    
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Zero();  // Gravity doesn't depend on velocity
    
    if (r_norm < 1.0) {
        return {da_dr, da_dv};  // Zero near singularity
    }
    
    // For gravity: a = -μ/r³ * r
    // ∂a/∂r = -μ/r³ * I + 3μ/r⁵ * (r ⊗ r)
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d outer = r * r.transpose();
    
    da_dr = (-GM_ / std::pow(r_norm, 3)) * I + 
            (3.0 * GM_ / std::pow(r_norm, 5)) * outer;
    
    return {da_dr, da_dv};
}

J2Gravity::J2Gravity(double GM, double J2, double Re)
    : GM_(GM), J2_(J2), Re_(Re)
{}

auto J2Gravity::compute_force(const ForceContext& ctx) const -> Eigen::Vector3d {
    double r_norm = ctx.position.norm();
    double r2 = r_norm * r_norm;
    double r5 = r2 * r2 * r_norm;

    // Point-mass term
    Eigen::Vector3d a = -GM_ / (r_norm * r2) * ctx.position;

    // J2 perturbation
    double z_over_r = ctx.position(2) / r_norm;
    double z_over_r2 = z_over_r * z_over_r;
    Eigen::Vector3d a_j2;
    a_j2 << ctx.position(0) * (1.0 - 5.0 * z_over_r2),
            ctx.position(1) * (1.0 - 5.0 * z_over_r2),
            ctx.position(2) * (3.0 - 5.0 * z_over_r2);
    a_j2 *= 1.5 * GM_ * J2_ * Re_ * Re_ / r5;

    return a + a_j2;
}

auto J2Gravity::compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {
    Eigen::Vector3d r = ctx.position;
    double r_norm = r.norm();
    
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Zero();  // Gravity doesn't depend on velocity
    
    if (r_norm < 1.0) {
        return {da_dr, da_dv};  // Zero near singularity
    }
    
    double x = r(0), y = r(1), z = r(2);
    double r2 = r_norm * r_norm;
    double r3 = r2 * r_norm;
    double r5 = r2 * r3;
    double r7 = r5 * r2;
    
    // Point-mass Jacobian: ∂a/∂r = -μ/r³ * I + 3μ/r⁵ * (r ⊗ r)
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d outer = r * r.transpose();
    Eigen::Matrix3d da_dr_pm = (-GM_ / r3) * I + (3.0 * GM_ / r5) * outer;
    
    // J2 perturbation Jacobian
    double mu_J2_Re2 = GM_ * J2_ * Re_ * Re_;
    double z2 = z * z;
    
    // Precompute common terms
    double coef1 = 1.5 * mu_J2_Re2 / r7;
    double coef2 = 7.5 * mu_J2_Re2 / (r7 * r2);
    
    // Compute partial derivatives
    da_dr(0, 0) = coef1 * (r2 - 6*z2) - coef2 * x*x * (r2 - 7*z2);
    da_dr(0, 1) = -coef2 * x*y * (r2 - 7*z2);
    da_dr(0, 2) = coef1 * 10*x*z - coef2 * x*z * (r2 - 7*z2);
    
    da_dr(1, 0) = da_dr(0, 1);
    da_dr(1, 1) = coef1 * (r2 - 6*z2) - coef2 * y*y * (r2 - 7*z2);
    da_dr(1, 2) = coef1 * 10*y*z - coef2 * y*z * (r2 - 7*z2);
    
    da_dr(2, 0) = da_dr(0, 2);
    da_dr(2, 1) = da_dr(1, 2);
    da_dr(2, 2) = coef1 * (3*r2 - 4*z2) - coef2 * z*z * (r2 - 7*z2);
    
    da_dr += da_dr_pm;
    
    return {da_dr, da_dv};
}

} // namespace dynamics
