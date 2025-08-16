#pragma once

#include <Eigen/Dense>

/// @brief Base class defining how a system evolves by specifying the derivative of the state vector with respect to time
///        based on the physical laws governing the system.
class Dynamics {
public:
    /// @brief virtual dtor
    virtual ~Dynamics() = default;
    
    /// @brief Computes the time derivative of the state vector at a given time.
    /// @param t Current time (in seconds).
    /// @param state Current state vector (e.g., position and velocity components).
    /// @return The derivative of the state vector (e.g., velocities and accelerations).
    virtual Eigen::VectorXd derivative(double t, const Eigen::VectorXd& state) const = 0;
};
