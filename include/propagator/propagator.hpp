#pragma once

#include <dynamics/ballistic3d.hpp>
#include <integrator/rk4.hpp>
#include <vector>
#include <memory>


class Propagator {
public:
    Propagator(std::shared_ptr<Dynamics> dynamics, std::shared_ptr<Integrator> integrator, double timestep)
     : dynamics_(dynamics), integrator_(integrator), timestep_(timestep)
    {}

    /// @brief propagate state to specific time
    /// @todo update this to take a termination predicate so user can decide when prop is complete
    ///
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    /// @param tf time to propagate to
    ///
    /// @return propagated states
    std::vector<std::pair<double, Eigen::VectorXd>> propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const
    {
        std::vector<std::pair<double, Eigen::VectorXd>> trajectory;
        double t = t0;
        Eigen::VectorXd state = initial_state;
        trajectory.emplace_back(t, state);

        while (t < tf) {
            double step_dt = std::min(timestep_, tf - t);
            state = integrator_->step(t, state, step_dt, *dynamics_);
            t += step_dt;
            trajectory.emplace_back(t, state);
        }

        return trajectory;
    }

    /// @brief propagate state to ground impact
    /// @todo update this to take a termination predicate so user can decide when prop is complete
    ///
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    ///
    /// @return propagated states
    std::vector<std::pair<double, Eigen::VectorXd>> propagate_to_impact(double t0, const Eigen::VectorXd& initial_state) const
    {
        double earth_radius = 6371000; // (m)
        std::vector<std::pair<double, Eigen::VectorXd>> trajectory;
        double t = t0;
        Eigen::VectorXd state = initial_state;
        trajectory.emplace_back(t, state);

        while (state.head(3).norm() > earth_radius) {
            double step_dt = timestep_;
            state = integrator_->step(t, state, step_dt, *dynamics_);
            t += step_dt;
            trajectory.emplace_back(t, state);
        }

        return trajectory;
    }

private:
    /// @brief underlying system dynamics
    std::shared_ptr<Dynamics> dynamics_;
    /// @brief underlying integrator to use for propagation
    std::shared_ptr<Integrator> integrator_;
    /// @brief timestep to use for propagation
    double timestep_;
};
