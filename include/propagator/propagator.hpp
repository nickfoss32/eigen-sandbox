#pragma once

#include <vector>
#include <memory>

#include "dynamics.hpp"
#include "integrator.hpp"
#include "coord_transforms.hpp"


namespace propagator {
class Propagator {
public:
    Propagator(
        std::shared_ptr<dynamics::IDynamics> dynamics,
        std::shared_ptr<integrator::Integrator> integrator,
        double timestep,
        dynamics::CoordinateFrame frame,
        std::shared_ptr<transforms::CoordTransforms> transforms
    );

    /// @brief propagate state to specific time
    /// @todo update this to take a termination predicate so user can decide when prop is complete
    ///
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    /// @param tf time to propagate to
    ///
    /// @return propagated states
    auto propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const -> std::vector<std::pair<double, Eigen::VectorXd>>;

    /// @brief propagate state to ground impact
    /// @todo update this to take a termination predicate so user can decide when prop is complete
    ///
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    ///
    /// @return propagated states
    auto propagate_to_impact(double t0, const Eigen::VectorXd& initial_state) const -> std::vector<std::pair<double, Eigen::VectorXd>>;

private:
    /// @brief underlying system dynamics
    std::shared_ptr<dynamics::IDynamics> dynamics_;
    /// @brief underlying integrator to use for propagation
    std::shared_ptr<integrator::Integrator> integrator_;
    /// @brief timestep to use for propagation
    double timestep_;
    /// @brief Coordinate frame to use
    dynamics::CoordinateFrame coordinateFrame_;
    /// @brief coordinate transforms
    std::shared_ptr<transforms::CoordTransforms> coordTransforms_;
};
} // namespace propagator
