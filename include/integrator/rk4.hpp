#pragma once

#include <Eigen/Dense>
#include "integrator/integrator.hpp"

class RK4Integrator : public Integrator {
public:
    Eigen::VectorXd step(double t, const Eigen::VectorXd& state, double dt, const Dynamics& dyn) const override {
        Eigen::VectorXd k1 = dyn.derivative(t, state);
        Eigen::VectorXd k2 = dyn.derivative(t + dt / 2.0, state + (dt / 2.0) * k1);
        Eigen::VectorXd k3 = dyn.derivative(t + dt / 2.0, state + (dt / 2.0) * k2);
        Eigen::VectorXd k4 = dyn.derivative(t + dt, state + dt * k3);
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
};
