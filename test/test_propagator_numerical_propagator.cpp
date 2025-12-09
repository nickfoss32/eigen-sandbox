#include <gtest/gtest.h>

#include "propagator/numerical_propagator.hpp"
#include "dynamics/dynamics.hpp"
#include "integrator/integrator.hpp"

#include <Eigen/Dense>

#include <memory>
#include <cmath>

using namespace propagator;
using namespace dynamics;
using namespace integrator;

// Mock numerical dynamics: dx/dt = x (exponential growth)
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

// Mock numerical dynamics: dx/dt = 1 (constant velocity)
class ConstantVelocityDynamics : public IDynamics {
public:
    explicit ConstantVelocityDynamics(double velocity = 1.0) 
        : velocity_(velocity) {}
    
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return Eigen::VectorXd::Constant(state.size(), velocity_);
    }
    
    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Zero(state.size(), state.size());
    }

    auto get_state_dimension() const -> int override {
        return 1;
    }
    
private:
    double velocity_;
};

// Mock numerical dynamics: dx/dt = 0 (no change)
class StaticDynamics : public IDynamics {
public:
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return Eigen::VectorXd::Zero(state.size());
    }

    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Zero(state.size(), state.size());
    }

    auto get_state_dimension() const -> int override {
        return 1;
    }
};

// Mock numerical dynamics: simple harmonic oscillator
// State: [position, velocity]
// dx/dt = v, dv/dt = -omega^2 * x
class SimpleHarmonicOscillator : public IDynamics {
public:
    explicit SimpleHarmonicOscillator(double omega = 1.0) : omega_(omega) {}
    
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        Eigen::VectorXd deriv(2);
        deriv(0) = state(1);                    // dx/dt = v
        deriv(1) = -omega_ * omega_ * state(0); // dv/dt = -omega^2 * x
        return deriv;
    }
    
    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        Eigen::MatrixXd J(2, 2);
        J(0, 0) = 0.0;            J(0, 1) = 1.0;
        J(1, 0) = -omega_ * omega_; J(1, 1) = 0.0;
        return J;
    }

    auto get_state_dimension() const -> int override {
        return 2;
    }

private:
    double omega_;
};

// Mock integrator (simple Euler for testing)
class EulerIntegrator : public IIntegrator {
public:
    Eigen::VectorXd step(double t, const Eigen::VectorXd& state, double dt, const IDynamics& dyn) const override {
        return state + dt * dyn.compute_dynamics(t, state);
    }
};

// Mock integrator that just returns the state (no change)
class IdentityIntegrator : public IIntegrator {
public:
    Eigen::VectorXd step(double t, const Eigen::VectorXd& state, double dt, const IDynamics& dyn) const override {
        return state;
    }
};

// Test fixture
class NumericalPropagatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        exponential_dynamics_ = std::make_shared<ExponentialDynamics>();
        constant_vel_dynamics_ = std::make_shared<ConstantVelocityDynamics>(2.0);
        static_dynamics_ = std::make_shared<StaticDynamics>();
        oscillator_dynamics_ = std::make_shared<SimpleHarmonicOscillator>(1.0);
        euler_integrator_ = std::make_shared<EulerIntegrator>();
        identity_integrator_ = std::make_shared<IdentityIntegrator>();
    }
    
    std::shared_ptr<ExponentialDynamics> exponential_dynamics_;
    std::shared_ptr<ConstantVelocityDynamics> constant_vel_dynamics_;
    std::shared_ptr<StaticDynamics> static_dynamics_;
    std::shared_ptr<SimpleHarmonicOscillator> oscillator_dynamics_;
    std::shared_ptr<EulerIntegrator> euler_integrator_;
    std::shared_ptr<IdentityIntegrator> identity_integrator_;
};

// Test basic propagation
TEST_F(NumericalPropagatorTest, BasicPropagation) {
    double timestep = 0.1;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // Should have multiple points
    ASSERT_GT(trajectory.size(), 1);
    
    // Check initial state
    EXPECT_DOUBLE_EQ(trajectory[0].first, 0.0);
    EXPECT_DOUBLE_EQ(trajectory[0].second(0), 1.0);
    
    // Check final time
    EXPECT_DOUBLE_EQ(trajectory.back().first, 1.0);
}

// Test trajectory time points
TEST_F(NumericalPropagatorTest, TrajectoryTimePoints) {
    double timestep = 0.2;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // Should have points at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    ASSERT_EQ(trajectory.size(), 6);
    
    EXPECT_DOUBLE_EQ(trajectory[0].first, 0.0);
    EXPECT_DOUBLE_EQ(trajectory[1].first, 0.2);
    EXPECT_DOUBLE_EQ(trajectory[2].first, 0.4);
    EXPECT_DOUBLE_EQ(trajectory[3].first, 0.6);
    EXPECT_DOUBLE_EQ(trajectory[4].first, 0.8);
    EXPECT_DOUBLE_EQ(trajectory[5].first, 1.0);
}

// Test with constant velocity dynamics
TEST_F(NumericalPropagatorTest, ConstantVelocityPropagation) {
    double timestep = 0.1;
    NumericalPropagator propagator(constant_vel_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 0.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // With Euler and constant velocity 2.0, after 1.0 seconds: x = 0 + 2.0*1.0 = 2.0
    EXPECT_NEAR(trajectory.back().second(0), 2.0, 0.01);
}

// Test with static dynamics (no change)
TEST_F(NumericalPropagatorTest, StaticDynamics) {
    double timestep = 0.1;
    NumericalPropagator propagator(static_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(3);
    initial_state << 1.0, 2.0, 3.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // State should not change
    EXPECT_EQ(trajectory.back().second, initial_state);
}

// Test backward propagation
TEST_F(NumericalPropagatorTest, BackwardPropagation) {
    double timestep = 0.1;
    NumericalPropagator propagator(constant_vel_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 10.0;
    
    // Propagate backward from t=1 to t=0
    auto trajectory = propagator.propagate(1.0, initial_state, 0.0);
    
    ASSERT_GT(trajectory.size(), 1);
    EXPECT_DOUBLE_EQ(trajectory[0].first, 1.0);
    EXPECT_DOUBLE_EQ(trajectory.back().first, 0.0);
    
    // Going backward with velocity 2.0 for 1 second: x = 10 - 2*1 = 8.0
    EXPECT_NEAR(trajectory.back().second(0), 8.0, 0.01);
}

// Test zero time interval
TEST_F(NumericalPropagatorTest, ZeroTimeInterval) {
    double timestep = 0.1;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(2);
    initial_state << 5.0, 10.0;
    
    auto trajectory = propagator.propagate(5.0, initial_state, 5.0);
    
    // Should have just the initial point
    ASSERT_EQ(trajectory.size(), 1);
    EXPECT_DOUBLE_EQ(trajectory[0].first, 5.0);
    EXPECT_EQ(trajectory[0].second, initial_state);
}

// Test timestep larger than time interval
TEST_F(NumericalPropagatorTest, LargeTimestep) {
    double timestep = 2.0;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    // Timestep is larger than interval, should take one step to final time
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    EXPECT_DOUBLE_EQ(trajectory[0].first, 0.0);
    EXPECT_DOUBLE_EQ(trajectory[1].first, 1.0);
}

// Test non-integer number of timesteps
TEST_F(NumericalPropagatorTest, NonIntegerTimesteps) {
    double timestep = 0.3;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    // 1.0 / 0.3 = 3.33... steps
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // Should have points at 0.0, 0.3, 0.6, 0.9, 1.0
    ASSERT_EQ(trajectory.size(), 5);
    EXPECT_DOUBLE_EQ(trajectory.back().first, 1.0);
}

// Test with multi-dimensional state
TEST_F(NumericalPropagatorTest, MultiDimensionalState) {
    double timestep = 0.1;
    NumericalPropagator propagator(constant_vel_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(5);
    initial_state << 0.0, 1.0, 2.0, 3.0, 4.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    ASSERT_EQ(trajectory.back().second.size(), 5);
    
    // Each component should increase by 2.0
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(trajectory.back().second(i), initial_state(i) + 2.0, 0.01);
    }
}

// Test initial state is preserved
TEST_F(NumericalPropagatorTest, InitialStatePreserved) {
    double timestep = 0.1;
    NumericalPropagator propagator(static_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(3);
    initial_state << 1.5, 2.5, 3.5;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // First point should be exact copy of initial state
    EXPECT_EQ(trajectory[0].second, initial_state);
}

// Test with negative initial time
TEST_F(NumericalPropagatorTest, NegativeInitialTime) {
    double timestep = 0.1;
    NumericalPropagator propagator(constant_vel_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 0.0;
    
    auto trajectory = propagator.propagate(-1.0, initial_state, 0.0);
    
    EXPECT_DOUBLE_EQ(trajectory[0].first, -1.0);
    EXPECT_DOUBLE_EQ(trajectory.back().first, 0.0);
    EXPECT_NEAR(trajectory.back().second(0), 2.0, 0.01);
}

// Test with large time interval
TEST_F(NumericalPropagatorTest, LargeTimeInterval) {
    double timestep = 1.0;
    NumericalPropagator propagator(constant_vel_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 0.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 100.0);
    
    ASSERT_EQ(trajectory.size(), 101); // 0, 1, 2, ..., 100
    EXPECT_NEAR(trajectory.back().second(0), 200.0, 1.0);
}

// Test state vector size is preserved
TEST_F(NumericalPropagatorTest, StateVectorSizePreserved) {
    double timestep = 0.1;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(10);
    initial_state.setOnes();
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    for (const auto& [t, state] : trajectory) {
        EXPECT_EQ(state.size(), 10);
    }
}

// Test trajectory is time-ordered
TEST_F(NumericalPropagatorTest, TrajectoryTimeOrdered) {
    double timestep = 0.1;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    // Forward
    auto traj_forward = propagator.propagate(0.0, initial_state, 1.0);
    for (size_t i = 1; i < traj_forward.size(); ++i) {
        EXPECT_GT(traj_forward[i].first, traj_forward[i-1].first);
    }
    
    // Backward
    auto traj_backward = propagator.propagate(1.0, initial_state, 0.0);
    for (size_t i = 1; i < traj_backward.size(); ++i) {
        EXPECT_LT(traj_backward[i].first, traj_backward[i-1].first);
    }
}

// Test with very small timestep
TEST_F(NumericalPropagatorTest, VerySmallTimestep) {
    double timestep = 0.001;
    NumericalPropagator propagator(constant_vel_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 0.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 0.1);
    
    ASSERT_EQ(trajectory.size(), 101); // 0.000, 0.001, ..., 0.100
    EXPECT_NEAR(trajectory.back().second(0), 0.2, 0.001);
}

// Test harmonic oscillator energy conservation (qualitative)
TEST_F(NumericalPropagatorTest, HarmonicOscillator) {
    double timestep = 0.01;
    NumericalPropagator propagator(oscillator_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(2);
    initial_state << 1.0, 0.0; // Start at max displacement
    
    auto trajectory = propagator.propagate(0.0, initial_state, 6.28); // One period
    
    // Check that oscillation occurs (position goes negative)
    bool found_negative = false;
    for (const auto& [t, state] : trajectory) {
        if (state(0) < 0.0) {
            found_negative = true;
            break;
        }
    }
    EXPECT_TRUE(found_negative);
}

// Test with very small state values
TEST_F(NumericalPropagatorTest, SmallStateValues) {
    double timestep = 0.1;
    NumericalPropagator propagator(static_dynamics_, euler_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1e-15;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // State should remain small
    EXPECT_NEAR(trajectory.back().second(0), 1e-15, 1e-20);
}

// Test trajectory size calculation
TEST_F(NumericalPropagatorTest, TrajectorySize) {
    double timestep = 0.25;
    NumericalPropagator propagator(static_dynamics_, identity_integrator_, timestep);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // 0.0, 0.25, 0.5, 0.75, 1.0 = 5 points
    EXPECT_EQ(trajectory.size(), 5);
}
