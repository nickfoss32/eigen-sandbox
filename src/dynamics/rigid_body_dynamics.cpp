#include "dynamics/rigid_body_dynamics.hpp"
#include "dynamics/atmosphere.hpp"

namespace dynamics {

RigidBodyDynamics6DOF::RigidBodyDynamics6DOF(
    std::vector<std::shared_ptr<IForce>> forces,
    std::vector<std::shared_ptr<ITorque>> torques,
    const Eigen::Matrix3d& inertia,
    double mass
) : forces_(std::move(forces)),
    torques_(std::move(torques)),
    inertia_(inertia),
    inertia_inv_(inertia.inverse()),
    mass_(mass)
{}

auto RigidBodyDynamics6DOF::compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd {
    // Extract state components
    Eigen::Vector3d position = state.segment<3>(0);
    Eigen::Vector3d velocity = state.segment<3>(3);
    Eigen::Quaterniond quat = Eigen::Map<const Eigen::Quaterniond>(state.segment<4>(6).data()); // x, y, z, w
    quat.normalize(); // Ensure unit quaternion
    Eigen::Vector3d omega = state.segment<3>(10); // angular velocity in body frame

    // Build force context
    ForceContext force_ctx;
    force_ctx.t = t;
    force_ctx.position = position;
    force_ctx.velocity = velocity;

    // Sum all accelerations (in inertial frame)
    Eigen::Vector3d total_acceleration = Eigen::Vector3d::Zero();
    for (const auto& force : forces_) {
        total_acceleration += force->compute_acceleration(force_ctx) * mass_;
    }
    Eigen::Vector3d acceleration = total_acceleration / mass_;

    // Build torque context
    TorqueContext torque_ctx;
    torque_ctx.t = t;
    torque_ctx.position = position;
    torque_ctx.velocity = velocity;
    torque_ctx.orientation = quat;
    torque_ctx.angular_velocity = omega;

    // Sum all torques (in body frame)
    Eigen::Vector3d total_torque = Eigen::Vector3d::Zero();
    for (const auto& torque : torques_) {
        total_torque += torque->compute_torque(torque_ctx);
    }

    // Euler's rotation equation: I * dω/dt = τ - ω × (I * ω)
    Eigen::Vector3d omega_dot = inertia_inv_ * (total_torque - omega.cross(inertia_ * omega));

    // Quaternion derivative: dq/dt = 0.5 * q * [0, ω]
    Eigen::Quaterniond omega_quat(0.0, omega.x(), omega.y(), omega.z());
    Eigen::Quaterniond quat_dot_quat = quat * omega_quat;
    quat_dot_quat.coeffs() *= 0.5;

    // Build derivative vector
    Eigen::VectorXd state_dot(13);
    state_dot.segment<3>(0) = velocity;                    // dx/dt = v
    state_dot.segment<3>(3) = acceleration;                // dv/dt = a
    state_dot.segment<4>(6) = quat_dot_quat.coeffs();      // dq/dt (x, y, z, w)
    state_dot.segment<3>(10) = omega_dot;                  // dω/dt

    return state_dot;
}

auto RigidBodyDynamics6DOF::compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd {
    // State vector: x = [r(3), v(3), q(4), ω(3)]ᵀ  (13D)
    // Derivative:   ẋ = [v, a, q̇, ω̇]ᵀ
    //
    // Jacobian F = ∂ẋ/∂x is 13×13:
    //
    //     | ∂v/∂r   ∂v/∂v   ∂v/∂q   ∂v/∂ω |   | 0   I   0   0 |
    //     | ∂a/∂r   ∂a/∂v   ∂a/∂q   ∂a/∂ω |   | *   *   0   0 |
    // F = | ∂q̇/∂r   ∂q̇/∂v   ∂q̇/∂q   ∂q̇/∂ω | = | 0   0   *   * |
    //     | ∂ω̇/∂r   ∂ω̇/∂v   ∂ω̇/∂q   ∂ω̇/∂ω |   | *   *   *   * |
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(13, 13);
    
    // Extract state components
    Eigen::Vector3d position = state.segment<3>(0);
    Eigen::Vector3d velocity = state.segment<3>(3);
    Eigen::Quaterniond quat = Eigen::Map<const Eigen::Quaterniond>(state.segment<4>(6).data());
    quat.normalize();
    Eigen::Vector3d omega = state.segment<3>(10);
    
    // ===================================================================
    // TRANSLATIONAL DYNAMICS: v̇ = a = Σf_i(r, v) / m
    // ===================================================================
    
    // Block (0:3, 3:6): ∂v/∂v = I
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    
    // Blocks (3:6, 0:3) and (3:6, 3:6): ∂a/∂r and ∂a/∂v
    // Sum force Jacobians from all forces
    ForceContext force_ctx{t, position, velocity};
    
    Eigen::Matrix3d da_dr_total = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d da_dv_total = Eigen::Matrix3d::Zero();
    
    for (const auto& force : forces_) {
        auto [da_dr, da_dv] = force->compute_jacobian(force_ctx);
        da_dr_total += da_dr;
        da_dv_total += da_dv;
    }
    
    F.block<3, 3>(3, 0) = da_dr_total;  // ∂a/∂r
    F.block<3, 3>(3, 3) = da_dv_total;  // ∂a/∂v
    
    // ===================================================================
    // ROTATIONAL KINEMATICS: q̇ = ½·Ω(ω)·q
    // ===================================================================
    
    // Block (6:10, 6:10): ∂q̇/∂q
    // q̇ = 0.5 * q * [0, ω]ᵀ  (quaternion multiplication)
    // This gives: ∂q̇/∂q = 0.5 * Ω(ω)
    Eigen::Matrix<double, 4, 4> omega_matrix;
    omega_matrix << 0.0,      -omega.x(), -omega.y(), -omega.z(),
                    omega.x(),  0.0,       omega.z(), -omega.y(),
                    omega.y(), -omega.z(),  0.0,       omega.x(),
                    omega.z(),  omega.y(), -omega.x(),  0.0;
    
    F.block<4, 4>(6, 6) = 0.5 * omega_matrix;
    
    // Block (6:10, 10:13): ∂q̇/∂ω
    // For q̇ = 0.5 * q ⊗ [0, ω_x, ω_y, ω_z]:
    // The result in (x, y, z, w) component order is:
    Eigen::Matrix<double, 4, 3> dq_dot_domega;
    dq_dot_domega <<  quat.w(),  quat.z(), -quat.y(),   // ∂x/∂ω
                     -quat.z(),  quat.w(),  quat.x(),   // ∂y/∂ω
                      quat.y(), -quat.x(),  quat.w(),   // ∂z/∂ω
                     -quat.x(), -quat.y(), -quat.z();   // ∂w/∂ω
    
    F.block<4, 3>(6, 10) = 0.5 * dq_dot_domega;
    
    // ===================================================================
    // ROTATIONAL DYNAMICS: ω̇ = I⁻¹·(τ - ω × I·ω)
    // ===================================================================
    
    TorqueContext torque_ctx{t, position, velocity, quat, omega};
    
    // Sum torque Jacobians from all torques
    Eigen::Matrix3d dtau_dq_total = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dtau_domega_total = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dtau_dr_total = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dtau_dv_total = Eigen::Matrix3d::Zero();
    
    for (const auto& torque : torques_) {
        auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque->compute_jacobian(torque_ctx);
        dtau_dq_total += dtau_dq;
        dtau_domega_total += dtau_domega;
        dtau_dr_total += dtau_dr;
        dtau_dv_total += dtau_dv;
    }
    
    // ω̇ = I⁻¹·(τ(q,ω,r,v) - ω × I·ω)
    // ∂ω̇/∂r = I⁻¹·∂τ/∂r
    F.block<3, 3>(10, 0) = inertia_inv_ * dtau_dr_total;
    
    // ∂ω̇/∂v = I⁻¹·∂τ/∂v
    F.block<3, 3>(10, 3) = inertia_inv_ * dtau_dv_total;
    
    // ∂ω̇/∂q = I⁻¹·∂τ/∂q
    // Note: This is 3×4 but we're using 3-param representation in torque Jacobian
    // For now, treating q as 3-param error (will need conversion for full 4-param)
    // TODO: Proper handling of 4-parameter quaternion Jacobian
    F.block<3, 3>(10, 6) = inertia_inv_ * dtau_dq_total;  // Simplified: assumes 3×3
    
    // ∂ω̇/∂ω = I⁻¹·(∂τ/∂ω - ∂(ω × I·ω)/∂ω)
    Eigen::Vector3d I_omega = inertia_ * omega;
    
    // Skew-symmetric matrices
    Eigen::Matrix3d skew_omega;
    skew_omega << 0.0,       -omega.z(),  omega.y(),
                  omega.z(),  0.0,       -omega.x(),
                 -omega.y(),  omega.x(),  0.0;
    
    Eigen::Matrix3d skew_I_omega;
    skew_I_omega << 0.0,         -I_omega.z(),  I_omega.y(),
                    I_omega.z(),  0.0,         -I_omega.x(),
                   -I_omega.y(),  I_omega.x(),  0.0;
    
    // ∂(ω × I·ω)/∂ω = -[ω]×·I - [I·ω]×
    Eigen::Matrix3d d_gyro_domega = -skew_omega * inertia_ - skew_I_omega;
    
    F.block<3, 3>(10, 10) = inertia_inv_ * (dtau_domega_total - d_gyro_domega);
    
    return F;
}

auto RigidBodyDynamics6DOF::get_state_dimension() const -> int {
    return 13;
}

} // namespace dynamics
