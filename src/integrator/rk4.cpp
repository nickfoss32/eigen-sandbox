#include "integrator/rk4.hpp"

namespace integrator {
Eigen::VectorXd RK4Integrator::step(double t, const Eigen::VectorXd& state, double dt, const dynamics::IDynamics& dyn) const {
    Eigen::VectorXd k1 = dyn.compute_dynamics(t, state);
    Eigen::VectorXd k2 = dyn.compute_dynamics(t + dt / 2.0, state + (dt / 2.0) * k1);
    Eigen::VectorXd k3 = dyn.compute_dynamics(t + dt / 2.0, state + (dt / 2.0) * k2);
    Eigen::VectorXd k4 = dyn.compute_dynamics(t + dt, state + dt * k3);
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}
} // namespace integrator
