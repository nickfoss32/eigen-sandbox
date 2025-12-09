#pragma once

#include "dynamics/dynamics.hpp"
#include "dynamics/force.hpp"

#include <Eigen/Dense>

#include <memory>
#include <vector>

namespace dynamics {
/// @brief Point mass dynamics model with configurable force models
///
/// @details
/// This class implements translational dynamics for a point mass (particle)
/// in 3D space. The dynamics are defined by summing accelerations from
/// multiple force models. No rotational dynamics are included.
class PointMassDynamics : public IDynamics {
public:
    /// @brief Constructs a 3D point mass dynamics model
    /// @param forces Vector of force models acting on the body
    explicit PointMassDynamics(std::vector<std::shared_ptr<IForce>> forces);

    /// @brief Computes the time derivative of the state vector
    /// @param t Current time (seconds)
    /// @param state 6D state vector [x, y, z, vx, vy, vz] (m, m/s)
    /// @return 6D derivative vector [vx, vy, vz, ax, ay, az] (m/s, m/s²)
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override;

    /// @brief Compute Jacobian of dynamics: ∂f/∂x
    /// @param t Time
    /// @param state Current state vector
    /// @return Jacobian matrix ∂f/∂x
    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override;

    /// @brief Get dimension of state vector
    auto get_state_dimension() const -> int override;

private:
    std::vector<std::shared_ptr<IForce>> forces_; ///< Force models acting on the body
};
} // namespace dynamics
