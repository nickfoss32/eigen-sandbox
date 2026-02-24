#include "dynamics/forces/constant_acceleration_force.hpp"

namespace dynamics {

ConstantAccelerationForce::ConstantAccelerationForce(const Eigen::Vector3d& acceleration)
    : acceleration_(acceleration) {}

auto ConstantAccelerationForce::compute_acceleration(const ForceContext& /*ctx*/) const
    -> Eigen::Vector3d {
    return acceleration_;
}

auto ConstantAccelerationForce::compute_jacobian(const ForceContext& /*ctx*/) const
    -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {
    return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
}

} // namespace dynamics
