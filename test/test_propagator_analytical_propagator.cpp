#include <gtest/gtest.h>

#include "propagator/analytical_propagator.hpp"
#include "dynamics/dynamics.hpp"

#include <Eigen/Dense>

#include <memory>
#include <cmath>

using namespace propagator;
using namespace dynamics;

// Mock analytical dynamics: x(t) = x0 * e^t (exponential growth)
class ExponentialAnalyticalDynamics : public IDynamics {
public:
    auto solve_analytical(double t0, const Eigen::VectorXd& state0, double tf) const -> std::optional<Eigen::VectorXd> override {
        double dt = tf - t0;
        Eigen::VectorXd result = state0;
        result.array() *= std::exp(dt);
        return result;
    }

    auto has_analytical_solution() const -> bool override {
        return true;
    }

    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return state; // dx/dt = x
    }
    
    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Identity(state.size(), state.size());
    }
    
    auto get_state_dimension() const -> int override {
        return 6;
    };
};

// Mock analytical dynamics: x(t) = x0 + v * t (constant velocity)
class ConstantVelocityAnalyticalDynamics : public IDynamics {
public:
    explicit ConstantVelocityAnalyticalDynamics(double velocity = 1.0) 
        : velocity_(velocity) {}
    
    auto solve_analytical(double t0, const Eigen::VectorXd& state0, double tf) const -> std::optional<Eigen::VectorXd> override {
        double dt = tf - t0;
        Eigen::VectorXd result = state0;
        result.array() += velocity_ * dt;
        return result;
    }

    auto has_analytical_solution() const -> bool override {
        return true;
    }

    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return state; // dx/dt = x
    }
    
    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Identity(state.size(), state.size());
    }
    
    auto get_state_dimension() const -> int override {
        return 6;
    };

private:
    double velocity_;
};

// Mock analytical dynamics: identity (state doesn't change)
class IdentityAnalyticalDynamics : public IDynamics {
public:
    auto solve_analytical(double t0, const Eigen::VectorXd& state0, double tf) const -> std::optional<Eigen::VectorXd> override {
        return state0;
    }

    auto has_analytical_solution() const -> bool override {
        return true;
    }

    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return state; // dx/dt = x
    }
    
    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Identity(state.size(), state.size());
    }
    
    auto get_state_dimension() const -> int override {
        return 6;
    };
};

// Test fixture
class AnalyticalPropagatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        exponential_dynamics_ = std::make_shared<ExponentialAnalyticalDynamics>();
        constant_vel_dynamics_ = std::make_shared<ConstantVelocityAnalyticalDynamics>(2.0);
        identity_dynamics_ = std::make_shared<IdentityAnalyticalDynamics>();
    }
    
    std::shared_ptr<ExponentialAnalyticalDynamics> exponential_dynamics_;
    std::shared_ptr<ConstantVelocityAnalyticalDynamics> constant_vel_dynamics_;
    std::shared_ptr<IdentityAnalyticalDynamics> identity_dynamics_;
};

// Test basic propagation
TEST_F(AnalyticalPropagatorTest, BasicPropagation) {
    AnalyticalPropagator propagator(exponential_dynamics_);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    // Should have two points: initial and final
    ASSERT_EQ(trajectory.size(), 2);
    
    // Check initial state
    EXPECT_DOUBLE_EQ(trajectory[0].first, 0.0);
    EXPECT_DOUBLE_EQ(trajectory[0].second(0), 1.0);
    
    // Check final state: x(1) = 1.0 * e^1 = e
    EXPECT_DOUBLE_EQ(trajectory[1].first, 1.0);
    EXPECT_NEAR(trajectory[1].second(0), std::exp(1.0), 1e-10);
}

// Test trajectory contains correct time points
TEST_F(AnalyticalPropagatorTest, TrajectoryTimePoints) {
    AnalyticalPropagator propagator(identity_dynamics_);
    
    Eigen::VectorXd initial_state(3);
    initial_state << 1.0, 2.0, 3.0;
    
    double t0 = 5.0;
    double tf = 10.0;
    
    auto trajectory = propagator.propagate(t0, initial_state, tf);
    
    ASSERT_EQ(trajectory.size(), 2);
    EXPECT_DOUBLE_EQ(trajectory[0].first, t0);
    EXPECT_DOUBLE_EQ(trajectory[1].first, tf);
}

// Test with constant velocity dynamics
TEST_F(AnalyticalPropagatorTest, ConstantVelocityPropagation) {
    AnalyticalPropagator propagator(constant_vel_dynamics_);
    
    Eigen::VectorXd initial_state(2);
    initial_state << 0.0, 5.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 2.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    
    // x(2) = x0 + v*t = 0.0 + 2.0*2 = 4.0
    EXPECT_NEAR(trajectory[1].second(0), 4.0, 1e-10);
    // y(2) = y0 + v*t = 5.0 + 2.0*2 = 9.0
    EXPECT_NEAR(trajectory[1].second(1), 9.0, 1e-10);
}

// Test with identity dynamics (state doesn't change)
TEST_F(AnalyticalPropagatorTest, IdentityDynamics) {
    AnalyticalPropagator propagator(identity_dynamics_);
    
    Eigen::VectorXd initial_state(4);
    initial_state << 1.0, 2.0, 3.0, 4.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 100.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    
    // Final state should equal initial state
    EXPECT_EQ(trajectory[1].second, initial_state);
}

// Test backward propagation (t0 > tf)
TEST_F(AnalyticalPropagatorTest, BackwardPropagation) {
    AnalyticalPropagator propagator(exponential_dynamics_);
    
    Eigen::VectorXd initial_state(1);
    initial_state << std::exp(1.0); // e
    
    // Propagate backward from t=1 to t=0
    auto trajectory = propagator.propagate(1.0, initial_state, 0.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    
    // x(0) from x(1)=e should give x(0) = e * e^(-1) = 1.0
    EXPECT_NEAR(trajectory[1].second(0), 1.0, 1e-10);
}

// Test with zero time interval
TEST_F(AnalyticalPropagatorTest, ZeroTimeInterval) {
    AnalyticalPropagator propagator(identity_dynamics_);
    
    Eigen::VectorXd initial_state(2);
    initial_state << 5.0, 10.0;
    
    auto trajectory = propagator.propagate(5.0, initial_state, 5.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    
    // Both points should have same time and state
    EXPECT_DOUBLE_EQ(trajectory[0].first, 5.0);
    EXPECT_DOUBLE_EQ(trajectory[1].first, 5.0);
    EXPECT_EQ(trajectory[0].second, initial_state);
    EXPECT_EQ(trajectory[1].second, initial_state);
}

// Test with multi-dimensional state
TEST_F(AnalyticalPropagatorTest, MultiDimensionalState) {
    AnalyticalPropagator propagator(exponential_dynamics_);
    
    Eigen::VectorXd initial_state(5);
    initial_state << 1.0, 2.0, 3.0, 4.0, 5.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    ASSERT_EQ(trajectory[1].second.size(), 5);
    
    // Each component should be multiplied by e
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(trajectory[1].second(i), initial_state(i) * std::exp(1.0), 1e-10);
    }
}

// Test initial state is preserved in trajectory
TEST_F(AnalyticalPropagatorTest, InitialStatePreserved) {
    AnalyticalPropagator propagator(exponential_dynamics_);
    
    Eigen::VectorXd initial_state(3);
    initial_state << 1.5, 2.5, 3.5;
    
    auto trajectory = propagator.propagate(2.0, initial_state, 5.0);
    
    // First point should be exact copy of initial state
    EXPECT_EQ(trajectory[0].second, initial_state);
}

// Test with negative initial time
TEST_F(AnalyticalPropagatorTest, NegativeInitialTime) {
    AnalyticalPropagator propagator(constant_vel_dynamics_);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 10.0;
    
    auto trajectory = propagator.propagate(-5.0, initial_state, 0.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    
    // dt = 5, x(0) = 10 + 2*5 = 20
    EXPECT_NEAR(trajectory[1].second(0), 20.0, 1e-10);
}

// Test with large time interval
TEST_F(AnalyticalPropagatorTest, LargeTimeInterval) {
    AnalyticalPropagator propagator(constant_vel_dynamics_);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 0.0;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1000.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    
    // x(1000) = 0 + 2*1000 = 2000
    EXPECT_NEAR(trajectory[1].second(0), 2000.0, 1e-10);
}

// Test state vector size is preserved
TEST_F(AnalyticalPropagatorTest, StateVectorSizePreserved) {
    AnalyticalPropagator propagator(identity_dynamics_);
    
    Eigen::VectorXd initial_state(10);
    initial_state.setOnes();
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    EXPECT_EQ(trajectory[0].second.size(), 10);
    EXPECT_EQ(trajectory[1].second.size(), 10);
}

// Test trajectory ordering (time should be monotonic)
TEST_F(AnalyticalPropagatorTest, TrajectoryOrdering) {
    AnalyticalPropagator propagator(identity_dynamics_);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1.0;
    
    // Forward propagation
    auto traj_forward = propagator.propagate(0.0, initial_state, 10.0);
    EXPECT_LE(traj_forward[0].first, traj_forward[1].first);
    
    // Backward propagation
    auto traj_backward = propagator.propagate(10.0, initial_state, 0.0);
    EXPECT_GE(traj_backward[0].first, traj_backward[1].first);
}

// Test with very small state values
TEST_F(AnalyticalPropagatorTest, SmallStateValues) {
    AnalyticalPropagator propagator(exponential_dynamics_);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1e-10;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    EXPECT_NEAR(trajectory[1].second(0), 1e-10 * std::exp(1.0), 1e-20);
}

// Test with very large state values
TEST_F(AnalyticalPropagatorTest, LargeStateValues) {
    AnalyticalPropagator propagator(constant_vel_dynamics_);
    
    Eigen::VectorXd initial_state(1);
    initial_state << 1e10;
    
    auto trajectory = propagator.propagate(0.0, initial_state, 1.0);
    
    ASSERT_EQ(trajectory.size(), 2);
    EXPECT_NEAR(trajectory[1].second(0), 1e10 + 2.0, 1.0);
}
