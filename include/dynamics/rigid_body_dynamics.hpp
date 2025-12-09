#pragma once

#include "dynamics/dynamics.hpp"
#include "dynamics/force.hpp"
#include "dynamics/torque.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <vector>

namespace dynamics {

/// @brief 6DOF rigid body dynamics with configurable forces and torques
///
/// This class implements full 6-degree-of-freedom dynamics including both
/// translational motion (position/velocity) and rotational motion 
/// (orientation/angular velocity).
///
/// State vector (13 elements):
/// [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
/// - Position (m): x, y, z
/// - Velocity (m/s): vx, vy, vz
/// - Orientation (quaternion): qw, qx, qy, qz
/// - Angular velocity (rad/s, body frame): wx, wy, wz
class RigidBodyDynamics6DOF : public IDynamics {
public:
    /// @brief Constructs a 6DOF rigid body dynamics model
    /// @param forces Vector of force models acting on the body (inertial frame)
    /// @param torques Vector of torque models acting on the body (body frame)
    /// @param inertia Inertia tensor in body frame (kg·m²)
    /// @param mass Body mass (kg)
    RigidBodyDynamics6DOF(
        std::vector<std::shared_ptr<IForce>> forces,
        std::vector<std::shared_ptr<ITorque>> torques,
        const Eigen::Matrix3d& inertia,
        double mass
    );

    /// @brief Computes the time derivative of the state vector
    /// @param t Current time (seconds)
    /// @param state 13D state vector [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    /// @return 13D derivative vector
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override;

    /// @brief Compute Jacobian of dynamics: ∂f/∂x
    /// @param t Time
    /// @param state Current state vector
    /// @return Jacobian matrix ∂f/∂x
    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override;

    /// @brief Get dimension of state vector
    auto get_state_dimension() const -> int override;

private:
    std::vector<std::shared_ptr<IForce>> forces_;    ///< Force models (inertial frame)
    std::vector<std::shared_ptr<ITorque>> torques_;  ///< Torque models (body frame)
    Eigen::Matrix3d inertia_;                         ///< Inertia tensor (body frame)
    Eigen::Matrix3d inertia_inv_;                     ///< Inverse inertia tensor
    double mass_;                                     ///< Body mass (kg)
};

} // namespace dynamics
