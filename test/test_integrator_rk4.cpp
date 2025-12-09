#include <gtest/gtest.h>

#include "integrator/rk4.hpp"
#include "dynamics/dynamics.hpp"

#include <Eigen/Dense>

#include <cmath>

using namespace integrator;
using namespace dynamics;

// Mock dynamics for testing: dx/dt = x (exponential growth)
class ExponentialDynamics : public IDynamics {
public:
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return state; // dx/dt = x, solution is x(t) = x0 * e^t
    }

    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Identity(state.size(), state.size());
    }

    auto get_state_dimension() const -> int override {
        return 1;
    }
};

// Mock dynamics for testing: dx/dt = -x (exponential decay)
class DecayDynamics : public IDynamics {
public:
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return -state; // dx/dt = -x, solution is x(t) = x0 * e^(-t)
    }

    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return -Eigen::MatrixXd::Identity(state.size(), state.size());
    }

    auto get_state_dimension() const -> int override {
        return 1;
    }
};

// Mock dynamics for testing: dx/dt = 1 (constant velocity)
class ConstantVelocityDynamics : public IDynamics {
public:
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        Eigen::VectorXd deriv = Eigen::VectorXd::Ones(state.size());
        return deriv; // dx/dt = 1, solution is x(t) = x0 + t
    }

    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Zero(state.size(), state.size());
    }

    auto get_state_dimension() const -> int override {
        return 1;
    }
};

// Mock dynamics for testing: simple harmonic oscillator
// State: [position, velocity]
// dx/dt = v, dv/dt = -omega^2 * x
class SimpleHarmonicOscillator : public IDynamics {
public:
    explicit SimpleHarmonicOscillator(double omega = 1.0) : omega_(omega) {}
    
    Eigen::VectorXd compute_dynamics(double t, const Eigen::VectorXd& state) const override {
        Eigen::VectorXd deriv(2);
        deriv(0) = state(1);                    // dx/dt = v
        deriv(1) = -omega_ * omega_ * state(0); // dv/dt = -omega^2 * x
        return deriv;
    }

    Eigen::MatrixXd compute_jacobian(double t, const Eigen::VectorXd& state) const override {
        Eigen::MatrixXd J(2, 2);
        J(0, 0) = 0.0;            J(0, 1) = 1.0;
        J(1, 0) = -omega_ * omega_; J(1, 1) = 0.0;
        return J;
    }

    int get_state_dimension() const override {
        return 2;
    }
    
private:
    double omega_;
};

// Test fixture
class RK4IntegratorTest : public ::testing::Test {
protected:
    RK4Integrator integrator;
};

// Test with exponential growth dynamics
TEST_F(RK4IntegratorTest, ExponentialGrowth) {
    ExponentialDynamics dyn;
    Eigen::VectorXd state(1);
    state << 1.0;
    
    double t = 0.0;
    double dt = 0.1;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    // Analytical solution: x(0.1) = 1.0 * e^0.1 ≈ 1.10517
    double expected = std::exp(0.1);
    EXPECT_NEAR(new_state(0), expected, 1e-6);
}

// Test with exponential decay dynamics
TEST_F(RK4IntegratorTest, ExponentialDecay) {
    DecayDynamics dyn;
    Eigen::VectorXd state(1);
    state << 1.0;
    
    double t = 0.0;
    double dt = 0.1;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    // Analytical solution: x(0.1) = 1.0 * e^(-0.1) ≈ 0.904837
    double expected = std::exp(-0.1);
    EXPECT_NEAR(new_state(0), expected, 1e-6);
}

// Test with constant velocity dynamics
TEST_F(RK4IntegratorTest, ConstantVelocity) {
    ConstantVelocityDynamics dyn;
    Eigen::VectorXd state(1);
    state << 5.0;
    
    double t = 0.0;
    double dt = 0.5;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    // Analytical solution: x(0.5) = 5.0 + 0.5 = 5.5
    EXPECT_NEAR(new_state(0), 5.5, 1e-10);
}

// Test with simple harmonic oscillator
TEST_F(RK4IntegratorTest, SimpleHarmonicOscillator) {
    SimpleHarmonicOscillator dyn(1.0); // omega = 1
    Eigen::VectorXd state(2);
    state << 1.0, 0.0; // Initial position = 1, velocity = 0
    
    double t = 0.0;
    double dt = 0.1;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    // Analytical solution: x(t) = cos(t), v(t) = -sin(t)
    double expected_x = std::cos(0.1);
    double expected_v = -std::sin(0.1);
    
    EXPECT_NEAR(new_state(0), expected_x, 1e-6);
    EXPECT_NEAR(new_state(1), expected_v, 1e-6);
}

// Test with multiple dimensions
TEST_F(RK4IntegratorTest, MultiDimensionalState) {
    ExponentialDynamics dyn;
    Eigen::VectorXd state(3);
    state << 1.0, 2.0, 3.0;
    
    double t = 0.0;
    double dt = 0.1;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    double factor = std::exp(0.1);
    EXPECT_NEAR(new_state(0), 1.0 * factor, 1e-6);
    EXPECT_NEAR(new_state(1), 2.0 * factor, 1e-6);
    EXPECT_NEAR(new_state(2), 3.0 * factor, 1e-6);
}

// Test with zero time step
TEST_F(RK4IntegratorTest, ZeroTimeStep) {
    ExponentialDynamics dyn;
    Eigen::VectorXd state(1);
    state << 5.0;
    
    double t = 0.0;
    double dt = 0.0;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    // With dt = 0, state should not change
    EXPECT_DOUBLE_EQ(new_state(0), 5.0);
}

// Test with negative time step (backward integration)
TEST_F(RK4IntegratorTest, NegativeTimeStep) {
    ExponentialDynamics dyn;
    Eigen::VectorXd state(1);
    state << 1.0;
    
    double t = 1.0;
    double dt = -0.1;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    // Going backward: x(0.9) from x(1.0) = e^1
    // x(0.9) = e^0.9 = e^1 * e^(-0.1)
    double expected = std::exp(-0.1);
    EXPECT_NEAR(new_state(0), expected, 1e-6);
}

// Test RK4 order of accuracy
TEST_F(RK4IntegratorTest, OrderOfAccuracy) {
    ExponentialDynamics dyn;
    Eigen::VectorXd state(1);
    state << 1.0;
    
    double t = 0.0;
    double T = 1.0; // Final time
    
    // Test with different step sizes
    std::vector<double> step_sizes = {0.1, 0.05, 0.025};
    std::vector<double> errors;
    
    double exact_solution = std::exp(T);
    
    for (double dt : step_sizes) {
        Eigen::VectorXd current_state = state;
        double current_t = t;
        
        // Integrate to T
        while (current_t < T) {
            double actual_dt = std::min(dt, T - current_t);
            current_state = integrator.step(current_t, current_state, actual_dt, dyn);
            current_t += actual_dt;
        }
        
        double error = std::abs(current_state(0) - exact_solution);
        errors.push_back(error);
    }
    
    // RK4 is 4th order: error ~ dt^4
    // So error(dt/2) / error(dt) should be approximately 2^4 = 16
    double ratio1 = errors[0] / errors[1];
    double ratio2 = errors[1] / errors[2];
    
    EXPECT_NEAR(ratio1, 16.0, 2.0); // Allow some tolerance
    EXPECT_NEAR(ratio2, 16.0, 2.0);
}

// Test conservation properties with harmonic oscillator
TEST_F(RK4IntegratorTest, EnergyConservation) {
    SimpleHarmonicOscillator dyn(1.0);
    Eigen::VectorXd state(2);
    state << 1.0, 0.0; // Start at max displacement
    
    double t = 0.0;
    double dt = 0.01;
    
    // Initial energy: E = 0.5 * (v^2 + omega^2 * x^2) = 0.5
    double initial_energy = 0.5 * (state(1) * state(1) + state(0) * state(0));
    
    // Integrate for several periods
    Eigen::VectorXd current_state = state;
    for (int i = 0; i < 628; ++i) { // ~10 periods (2*pi * 10 / dt)
        current_state = integrator.step(t, current_state, dt, dyn);
        t += dt;
    }
    
    // Final energy
    double final_energy = 0.5 * (current_state(1) * current_state(1) + 
                                  current_state(0) * current_state(0));
    
    // Energy should be approximately conserved (RK4 has some drift)
    EXPECT_NEAR(final_energy, initial_energy, 0.01);
}

// Test state vector size preservation
TEST_F(RK4IntegratorTest, StateVectorSizePreserved) {
    ExponentialDynamics dyn;
    Eigen::VectorXd state(5);
    state << 1.0, 2.0, 3.0, 4.0, 5.0;
    
    double t = 0.0;
    double dt = 0.1;
    
    Eigen::VectorXd new_state = integrator.step(t, state, dt, dyn);
    
    EXPECT_EQ(new_state.size(), state.size());
}
