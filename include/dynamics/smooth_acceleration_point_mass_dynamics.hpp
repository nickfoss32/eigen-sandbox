#pragma once

#include "dynamics/dynamics.hpp"
#include "dynamics/forces/force.hpp"

#include <Eigen/Dense>

#include <functional>
#include <memory>
#include <vector>

namespace dynamics {

/// @brief 9-state point-mass dynamics with a first-order smooth acceleration state.
///
/// State layout:
/// [x, y, z, vx, vy, vz, ax_s, ay_s, az_s]
///
/// The smooth acceleration state follows:
/// a_s_dot = (a_eq(t) - a_s) / tau
///
/// and contributes directly to translational acceleration:
/// v_dot = a_forces(r, v, t) + a_s
class SmoothAccelerationPointMassDynamics : public IDynamics {
public:
    using EquilibriumAccelerationFunction = std::function<Eigen::Vector3d(double)>;

    SmoothAccelerationPointMassDynamics(
        std::vector<std::shared_ptr<IForce>> forces,
        double correlation_time_seconds,
        EquilibriumAccelerationFunction equilibrium_acceleration_function = {}
    );

    auto compute_dynamics(double t, const Eigen::VectorXd& state) const
        -> Eigen::VectorXd override;

    auto compute_jacobian(double t, const Eigen::VectorXd& state) const
        -> Eigen::MatrixXd override;

    auto get_state_dimension() const -> int override;

private:
    std::vector<std::shared_ptr<IForce>> forces_;
    double correlation_time_seconds_ = 0.0;
    EquilibriumAccelerationFunction equilibrium_acceleration_function_;
};

} // namespace dynamics
