#pragma once

#include <Eigen/Dense>
#include "dynamics/dynamics.hpp"

/// @brief Abstract base class for numerical integrators that compute the next state of a system
///        by solving the differential equations defined by a dynamics model.
class Integrator {
public:
    /// @brief Virtual destructor to ensure proper cleanup of derived classes.
    virtual ~Integrator() = default;

    /// @brief Computes the next state by integrating the state derivative over a time step.
    /// @param t Current time (in seconds).
    /// @param state Current state vector (e.g., position and velocity components).
    /// @param dt Time step for integration (in seconds).
    /// @param dyn Dynamics model providing the state derivative.
    /// @return The updated state vector after one integration step.
    virtual Eigen::VectorXd step(double t, const Eigen::VectorXd& state, double dt, const dynamics::IDynamics& dyn) const = 0;
};
