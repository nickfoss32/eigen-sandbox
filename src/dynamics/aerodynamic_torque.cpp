#include "dynamics/aerodynamic_torque.hpp"
#include "dynamics/atmosphere.hpp"

namespace dynamics {

AerodynamicTorque::AerodynamicTorque(
    const Eigen::Vector3d& center_of_pressure,
    const Eigen::Vector3d& center_of_mass,
    double drag_coefficient,
    double reference_area
) : cop_(center_of_pressure),
    com_(center_of_mass),
    cd_(drag_coefficient),
    area_(reference_area)
{}

auto AerodynamicTorque::compute_torque(const TorqueContext& ctx) const -> Eigen::Vector3d {
    // Get atmospheric density at current altitude
    double altitude = ctx.position.norm() - 6.371e6;  // Subtract Earth radius
    double rho = atmosphere::get_density_us76(altitude);
    
    if (rho < 1e-15) {
        return Eigen::Vector3d::Zero();
    }
    
    // Transform velocity to body frame
    Eigen::Vector3d v_body = ctx.orientation.inverse() * ctx.velocity;
    double v_mag = v_body.norm();
    
    if (v_mag < 1e-6) {
        return Eigen::Vector3d::Zero();
    }
    
    // Drag force in body frame
    Eigen::Vector3d v_hat = v_body.normalized();
    Eigen::Vector3d drag_force_body = -0.5 * rho * cd_ * area_ * v_mag * v_mag * v_hat;
    
    // Moment arm from COM to COP
    Eigen::Vector3d moment_arm = cop_ - com_;
    
    // Torque = r × F
    return moment_arm.cross(drag_force_body);
}

auto AerodynamicTorque::compute_jacobian(const TorqueContext& ctx) const -> std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d> {
    // Get atmospheric density at current altitude
    double altitude = ctx.position.norm() - 6.371e6;
    double rho = atmosphere::get_density_us76(altitude);
    
    // If density negligible, all Jacobians are zero
    if (rho < 1e-15) {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
                Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    
    // Transform velocity to body frame
    Eigen::Vector3d v_body = ctx.orientation.inverse() * ctx.velocity;
    double v_mag = v_body.norm();
    
    // If velocity negligible, all Jacobians are zero
    if (v_mag < 1e-6) {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
                Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    
    Eigen::Vector3d v_hat = v_body.normalized();
    Eigen::Vector3d moment_arm = cop_ - com_;
    
    // Drag force in body frame: F = -0.5·ρ·Cd·A·|v|²·v̂
    double drag_coeff = -0.5 * rho * cd_ * area_;
    Eigen::Vector3d drag_force_body = drag_coeff * v_mag * v_mag * v_hat;
    
    // ===================================================================
    // ∂τ/∂q: How torque changes with attitude perturbation
    // ===================================================================
    // τ = r_arm × F_drag
    // where F_drag = f(v_body) and v_body = Rᵀ(q)·v_inertial
    //
    // For small attitude error δθ:
    // v_body_perturbed ≈ (I + [δθ×])·v_body
    // ∂v_body/∂δθ = -[v_body×]
    
    Eigen::Matrix3d v_body_skew;
    v_body_skew <<          0.0, -v_body(2),  v_body(1),
                    v_body(2),          0.0, -v_body(0),
                   -v_body(1),  v_body(0),          0.0;
    
    Eigen::Matrix3d dv_body_dtheta = -v_body_skew;
    
    // ∂F_drag/∂v_body: How drag force changes with body-frame velocity
    // F = drag_coeff · |v|² · v̂
    // ∂F/∂v = drag_coeff · ∂(|v|²·v̂)/∂v
    //       = drag_coeff · (2|v|·v̂⊗v̂ + |v|²·∂v̂/∂v)
    // where ∂v̂/∂v = (1/|v|)·(I - v̂⊗v̂)
    
    Eigen::Matrix3d v_hat_outer = v_hat * v_hat.transpose();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d dv_hat_dv = (1.0 / v_mag) * (I - v_hat_outer);
    
    Eigen::Matrix3d dF_dv_body = drag_coeff * (2.0 * v_mag * v_hat_outer + v_mag * v_mag * dv_hat_dv);
    
    // ∂τ/∂q: Use chain rule through v_body
    // τ = r_arm × F, so ∂τ/∂F = [r_arm×]
    Eigen::Matrix3d r_arm_skew;
    r_arm_skew <<              0.0, -moment_arm(2),  moment_arm(1),
                   moment_arm(2),              0.0, -moment_arm(0),
                  -moment_arm(1),  moment_arm(0),              0.0;
    
    Eigen::Matrix3d dtau_dq = r_arm_skew * dF_dv_body * dv_body_dtheta;
    
    // ===================================================================
    // ∂τ/∂ω: How torque changes with angular velocity
    // ===================================================================
    // In this simple model, torque doesn't directly depend on ω
    // (A more sophisticated model might include apparent wind from rotation)
    Eigen::Matrix3d dtau_domega = Eigen::Matrix3d::Zero();
    
    // ===================================================================
    // ∂τ/∂r: How torque changes with position (altitude affects density)
    // ===================================================================
    // τ = r_arm × F_drag
    // where F_drag = -0.5·ρ(h)·Cd·A·|v|²·v̂
    // and h = |r| - R_earth
    //
    // ∂τ/∂r = r_arm × (∂F_drag/∂ρ)·(∂ρ/∂h)·(∂h/∂r)
    
    double r_mag = ctx.position.norm();
    Eigen::Vector3d r_hat = ctx.position / r_mag;
    
    // ∂h/∂r = ∂|r|/∂r = r̂
    Eigen::Matrix3d dh_dr = r_hat * Eigen::RowVector3d::Ones();  // Rank-1: each column is r̂
    
    // ∂ρ/∂h: Atmospheric density gradient (numerical differentiation)
    double dh = 1.0;  // 1 meter altitude change
    double rho_plus = atmosphere::get_density_us76(altitude + dh);
    double rho_minus = atmosphere::get_density_us76(altitude - dh);
    double drho_dh = (rho_plus - rho_minus) / (2.0 * dh);
    
    // ∂F_drag/∂ρ = -0.5·Cd·A·|v|²·v̂
    Eigen::Vector3d dF_drho = -0.5 * cd_ * area_ * v_mag * v_mag * v_hat;
    
    // Chain rule: ∂F/∂r = (∂F/∂ρ)·(∂ρ/∂h)·(∂h/∂r)
    //                    = (∂F/∂ρ)·(∂ρ/∂h)·r̂ᵀ
    // This is a rank-1 matrix: dF_drho ⊗ (drho_dh · r̂)
    Eigen::Matrix3d dF_dr = dF_drho * (drho_dh * r_hat).transpose();
    
    // ∂τ/∂r = [r_arm×]·(∂F/∂r)
    Eigen::Matrix3d dtau_dr = r_arm_skew * dF_dr;
    
    // ===================================================================
    // ∂τ/∂v: How torque changes with velocity (in inertial frame)
    // ===================================================================
    // τ = r_arm × F_drag(v_body)
    // where v_body = Rᵀ·v_inertial
    //
    // ∂τ/∂v_inertial = [r_arm×]·(∂F/∂v_body)·(∂v_body/∂v_inertial)
    // where ∂v_body/∂v_inertial = Rᵀ
    
    Eigen::Matrix3d R_body_to_inertial = ctx.orientation.toRotationMatrix();
    Eigen::Matrix3d R_inertial_to_body = R_body_to_inertial.transpose();
    
    Eigen::Matrix3d dv_body_dv_inertial = R_inertial_to_body;
    
    // Chain rule
    Eigen::Matrix3d dtau_dv = r_arm_skew * dF_dv_body * dv_body_dv_inertial;
    
    return {dtau_dq, dtau_domega, dtau_dr, dtau_dv};
}
} // namespace dynamics
