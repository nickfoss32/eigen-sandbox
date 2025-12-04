#pragma once

#include <Eigen/Dense>

#include <utility>
#include <vector>

namespace propagator {
/// @brief Interface for all propagators
class IPropagator {
public:
    /// @brief Virtual destructor
    virtual ~IPropagator() = default;

    /// @brief Propagate state to specific time
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    /// @param tf time to propagate to
    /// @return propagated states
    virtual auto propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const -> std::vector<std::pair<double, Eigen::VectorXd>> = 0;

    /// @brief Compute state transition Jacobian using numerical differentiation
    /// @param t0 Initial time
    /// @param state Initial state
    /// @param dt Time step
    /// @return State transition matrix Φ(t0+dt, t0)
    virtual auto compute_transition_jacobian(double t0, const Eigen::VectorXd& state, double dt) const -> Eigen::MatrixXd = 0;
};
} // namespace propagator
