#include "dynamics/smooth_acceleration_point_mass_dynamics.hpp"

#include <stdexcept>

namespace dynamics {

SmoothAccelerationPointMassDynamics::SmoothAccelerationPointMassDynamics(
    std::vector<std::shared_ptr<IForce>> forces,
    double correlation_time_seconds,
    EquilibriumAccelerationFunction equilibrium_acceleration_function
) : forces_(std::move(forces))
  , correlation_time_seconds_(correlation_time_seconds)
  , equilibrium_acceleration_function_(std::move(equilibrium_acceleration_function))
{
    if (correlation_time_seconds_ <= 0.0) {
        throw std::invalid_argument(
            "SmoothAccelerationPointMassDynamics: correlation time must be positive"
        );
    }
}

auto SmoothAccelerationPointMassDynamics::compute_dynamics(
    double t,
    const Eigen::VectorXd& state
) const -> Eigen::VectorXd {
    if (state.size() != 9) {
        throw std::invalid_argument(
            "SmoothAccelerationPointMassDynamics: state must have size 9"
        );
    }

    ForceContext ctx;
    ctx.t = t;
    ctx.position = state.head<3>();
    ctx.velocity = state.segment<3>(3);

    Eigen::Vector3d total_force_acceleration = Eigen::Vector3d::Zero();
    for (const auto& force : forces_) {
        total_force_acceleration += force->compute_acceleration(ctx);
    }

    const Eigen::Vector3d smooth_acceleration = state.segment<3>(6);
    const Eigen::Vector3d equilibrium_acceleration =
        equilibrium_acceleration_function_ ? equilibrium_acceleration_function_(t)
                                          : Eigen::Vector3d::Zero();

    Eigen::VectorXd state_dot(9);
    state_dot.head<3>() = ctx.velocity;
    state_dot.segment<3>(3) = total_force_acceleration + smooth_acceleration;
    state_dot.tail<3>() =
        (equilibrium_acceleration - smooth_acceleration) / correlation_time_seconds_;
    return state_dot;
}

auto SmoothAccelerationPointMassDynamics::compute_jacobian(
    double t,
    const Eigen::VectorXd& state
) const -> Eigen::MatrixXd {
    if (state.size() != 9) {
        throw std::invalid_argument(
            "SmoothAccelerationPointMassDynamics: state must have size 9"
        );
    }

    ForceContext ctx;
    ctx.t = t;
    ctx.position = state.head<3>();
    ctx.velocity = state.segment<3>(3);

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(9, 9);
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 6) =
        (-1.0 / correlation_time_seconds_) * Eigen::Matrix3d::Identity();

    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Zero();
    for (const auto& force : forces_) {
        const auto jacobian = force->compute_jacobian(ctx);
        da_dr += jacobian.first;
        da_dv += jacobian.second;
    }

    F.block<3, 3>(3, 0) = da_dr;
    F.block<3, 3>(3, 3) = da_dv;
    return F;
}

auto SmoothAccelerationPointMassDynamics::get_state_dimension() const -> int {
    return 9;
}

} // namespace dynamics
