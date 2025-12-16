#include "dynamics/point_mass_dynamics.hpp"

namespace dynamics {
PointMassDynamics::PointMassDynamics(std::vector<std::shared_ptr<IForce>> forces)
: forces_(std::move(forces))
{}

auto PointMassDynamics::compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd {
    // Build context
    ForceContext ctx;
    ctx.t = t;
    ctx.position = state.head<3>();
    ctx.velocity = state.tail<3>();

    // Sum all forces
    Eigen::Vector3d total_acceleration = Eigen::Vector3d::Zero();
    for (const auto& force : forces_) {
        total_acceleration += force->compute_acceleration(ctx);
    }

    // Build derivative [velocity, acceleration]
    Eigen::VectorXd state_dot(6);
    state_dot.head<3>() = ctx.velocity;
    state_dot.tail<3>() = total_acceleration;

    return state_dot;
}

auto PointMassDynamics::compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd
{
    // Build context
    ForceContext ctx;
    ctx.t = t;
    ctx.position = state.head<3>();
    ctx.velocity = state.tail<3>();

    // Initialize Jacobian
    // F = [ ∂ṙ/∂r   ∂ṙ/∂v ]
    //     [ ∂v̇/∂r   ∂v̇/∂v ]
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(6, 6);
    
    // Upper-right block: ∂ṙ/∂v = I (velocity affects position rate)
    F.block<3,3>(0, 3) = Eigen::Matrix3d::Identity();
    
    // Lower blocks: sum Jacobians from all forces
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Zero();  // ∂v̇/∂r
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Zero();  // ∂v̇/∂v
    
    for (const auto& force : forces_) {
        auto force_jac = force->compute_jacobian(ctx);
        da_dr += force_jac.first;   // ∂a/∂r
        da_dv += force_jac.second;  // ∂a/∂v
    }
    
    F.block<3,3>(3, 0) = da_dr;
    F.block<3,3>(3, 3) = da_dv;
    
    return F;
}

auto PointMassDynamics::get_state_dimension() const -> int {
    return 6;
}
} // namespace dynamics
