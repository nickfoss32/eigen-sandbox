#include "dynamics/fictitious_forces.hpp"

namespace dynamics {

FictitiousForces::FictitiousForces(const Eigen::Vector3d& omega)
: omega_(omega)
{}

auto FictitiousForces::compute_force(const ForceContext& ctx) const -> Eigen::Vector3d {
    // Coriolis acceleration: -2 * omega x v
    Eigen::Vector3d a_coriolis = -2.0 * omega_.cross(ctx.velocity);

    // Centrifugal acceleration: -omega x (omega x r)
    Eigen::Vector3d a_centrifugal = -omega_.cross(omega_.cross(ctx.position));

    return a_coriolis + a_centrifugal;
}

auto FictitiousForces::compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d>{
    // ∂a/∂r = -omega x (omega x I)
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d omega_cross = Eigen::Matrix3d::Zero();
    omega_cross(0, 1) = -omega_(2);
    omega_cross(0, 2) = omega_(1);
    omega_cross(1, 0) = omega_(2);
    omega_cross(1, 2) = -omega_(0);
    omega_cross(2, 0) = -omega_(1);
    omega_cross(2, 1) = omega_(0);

    Eigen::Matrix3d da_dr = -omega_cross * omega_cross;

    // ∂a/∂v = -2 * omega x I
    Eigen::Matrix3d da_dv = -2.0 * omega_cross;

    return {da_dr, da_dv};
}

} // namespace dynamics
