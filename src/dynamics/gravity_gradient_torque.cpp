#include "dynamics/gravity_gradient_torque.hpp"

namespace dynamics {

GravityGradientTorque::GravityGradientTorque(const Eigen::Matrix3d& inertia, double GM)
: inertia_(inertia), GM_(GM)
{}

auto GravityGradientTorque::compute_torque(const TorqueContext& ctx) const -> Eigen::Vector3d {
    double r = ctx.position.norm();
    if (r < 1e-6) {
        return Eigen::Vector3d::Zero();
    }
    
    // Nadir direction in body frame
    Eigen::Vector3d r_hat_body = ctx.orientation.inverse() * (-ctx.position.normalized());
    
    // Gravity gradient torque: τ = (3μ/r³) * r_hat × (I * r_hat)
    double factor = 3.0 * GM_ / (r * r * r);
    Eigen::Vector3d torque = factor * r_hat_body.cross(inertia_ * r_hat_body);
    
    return torque;
}

auto GravityGradientTorque::compute_jacobian(const TorqueContext& ctx) const -> std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d> {
    // Gravity gradient depends on both q and r
    // Transform position from inertial to body frame
    Eigen::Vector3d r_body = ctx.orientation.inverse() * ctx.position;
    double r_mag = r_body.norm();
    Eigen::Vector3d z_body = r_body.normalized();
    
    // ∂τ/∂ω = 0 (gravity gradient doesn't depend on angular velocity)
    Eigen::Matrix3d dtau_domega = Eigen::Matrix3d::Zero();
    
    // ∂τ/∂v = 0 (gravity gradient doesn't depend on velocity)
    Eigen::Matrix3d dtau_dv = Eigen::Matrix3d::Zero();
    
    // ===================================================================
    // ∂τ/∂q: How torque changes with small attitude perturbations
    // ===================================================================
    // Gravity gradient torque: τ = 3μ/r³ · (ẑ_body × I·ẑ_body)
    // where ẑ_body = R(q)ᵀ·r_inertial / |r|
    //
    // For small attitude error δθ (3-parameter rotation vector):
    // R_perturbed ≈ R(q)·(I + [δθ×])
    // where [δθ×] is the skew-symmetric matrix
    //
    // Strategy: Use chain rule
    // ∂τ/∂δθ = ∂τ/∂ẑ · ∂ẑ/∂δθ
    
    double coeff = 3.0 * GM_ / (r_mag * r_mag * r_mag);
    Eigen::Vector3d I_z = inertia_ * z_body;
    
    // ∂τ/∂ẑ: How torque changes when body-frame position vector changes
    // τ = coeff · (ẑ × I·ẑ)
    // ∂(ẑ × I·ẑ)/∂ẑ = [I·ẑ]× - [ẑ]×·I
    // where [v]× is the skew-symmetric matrix of v
    
    Eigen::Matrix3d z_skew;
    z_skew <<      0.0, -z_body(2),  z_body(1),
              z_body(2),       0.0, -z_body(0),
             -z_body(1),  z_body(0),       0.0;
    
    Eigen::Matrix3d I_z_skew;
    I_z_skew <<    0.0, -I_z(2),  I_z(1),
                I_z(2),     0.0, -I_z(0),
               -I_z(1),  I_z(0),     0.0;
    
    Eigen::Matrix3d dtau_dz = coeff * (I_z_skew - z_skew * inertia_);
    
    // ∂ẑ/∂δθ: How body-frame position changes with attitude perturbation
    // For small rotation δθ:
    // ẑ_perturbed = (I + [δθ×])·ẑ ≈ ẑ + δθ × ẑ = ẑ - ẑ × δθ
    // So: ∂ẑ/∂δθ = -[ẑ×]
    
    Eigen::Matrix3d dz_dtheta = -z_skew;
    
    // Chain rule: ∂τ/∂δθ = ∂τ/∂ẑ · ∂ẑ/∂δθ
    Eigen::Matrix3d dtau_dq = dtau_dz * dz_dtheta;
    
    // ===================================================================
    // ∂τ/∂r: How torque changes with position (orbital altitude matters!)
    // ===================================================================
    // τ = 3μ/r³ · (ẑ_body × I·ẑ_body)
    // where ẑ_body = Rᵀ(q)·r / |r|
    //
    // Two effects:
    // 1) Magnitude changes: 3μ/r³ → ∂(3μ/r³)/∂r
    // 2) Direction changes: ẑ_body = Rᵀ·r/|r| → ∂ẑ_body/∂r
    
    // Effect 1: ∂(3μ/r³)/∂r in inertial frame
    // d/dr(3μ/r³) = -9μ/r⁴ · dr/dr = -9μ/r⁵ · r
    double r2 = r_mag * r_mag;
    double r3 = r2 * r_mag;
    double r5 = r3 * r2;
    
    Eigen::Vector3d tau = coeff * z_body.cross(I_z);
    Eigen::Vector3d dcoeff_dr = -9.0 * GM_ / r5 * ctx.position;
    
    // First term: (∂coeff/∂r) · (ẑ × I·ẑ)
    // This is a rank-1 matrix: outer product of tau and dcoeff_dr
    Eigen::Matrix3d term1 = tau * dcoeff_dr.transpose();
    
    // Effect 2: ∂ẑ_body/∂r in inertial frame
    // ẑ_body = Rᵀ·r / |r|
    // ∂ẑ_body/∂r = Rᵀ · ∂(r/|r|)/∂r
    // where ∂(r/|r|)/∂r = (1/|r|)·(I - r̂⊗r̂)
    
    Eigen::Matrix3d R_body_to_inertial = ctx.orientation.toRotationMatrix();
    Eigen::Matrix3d R_inertial_to_body = R_body_to_inertial.transpose();
    
    Eigen::Vector3d r_hat = ctx.position / r_mag;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d r_hat_outer = r_hat * r_hat.transpose();
    
    Eigen::Matrix3d dr_normalized_dr = (1.0 / r_mag) * (I - r_hat_outer);
    Eigen::Matrix3d dz_body_dr = R_inertial_to_body * dr_normalized_dr;
    
    // Second term: coeff · ∂(ẑ × I·ẑ)/∂ẑ · ∂ẑ/∂r
    // We already have ∂(ẑ × I·ẑ)/∂ẑ = dtau_dz / coeff
    Eigen::Matrix3d term2 = coeff * (dtau_dz / coeff) * dz_body_dr;
    
    // Total: ∂τ/∂r
    Eigen::Matrix3d dtau_dr = term1 + term2;
    
    return {dtau_dq, dtau_domega, dtau_dr, dtau_dv};
}
} // namespace dynamics
