#include <gtest/gtest.h>

#include "dynamics/atmospheric_drag.hpp"
#include "dynamics/point_mass_dynamics.hpp"
#include "dynamics/rigid_body_dynamics.hpp"
#include "dynamics/gravity.hpp"

#include <Eigen/Dense>
#include <memory>
#include <cmath>

using namespace dynamics;

// ============================================================================
// TEST FIXTURE
// ============================================================================

class AtmosphericDragTest : public ::testing::Test {
protected:
    // Typical satellite parameters
    double mass_;
    double drag_coeff_;
    double ref_area_;
    double earth_radius_;
    
    AtmosphericDragTest()
        : mass_(100.0)           // 100 kg satellite
        , drag_coeff_(2.2)       // Typical for satellite
        , ref_area_(1.0)         // 1 m² cross-section
        , earth_radius_(6378137.0)  // WGS84 Earth radius
    {}
    
    // Helper to create force context
    ForceContext createContext(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, double t = 0.0) {
        ForceContext ctx;
        ctx.t = t;
        ctx.position = pos;
        ctx.velocity = vel;
        return ctx;
    }
};

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

TEST_F(AtmosphericDragTest, ConstructorParameters) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_, earth_radius_);
    
    // Should construct without error
    // Parameters are private, but we can test behavior
    ForceContext ctx = createContext(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),  // 400 km altitude
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::Vector3d accel = drag.compute_acceleration(ctx);
    EXPECT_TRUE(accel.allFinite());
}

TEST_F(AtmosphericDragTest, DefaultEarthRadius) {
    // Test default Earth radius constructor
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    ForceContext ctx = createContext(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::Vector3d accel = drag.compute_acceleration(ctx);
    EXPECT_TRUE(accel.allFinite());
}

TEST_F(AtmosphericDragTest, ZeroVelocity) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    ForceContext ctx = createContext(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d::Zero()
    );
    
    Eigen::Vector3d accel = drag.compute_acceleration(ctx);
    
    // No velocity means no drag
    EXPECT_TRUE(accel.isApprox(Eigen::Vector3d::Zero(), 1e-15));
}

TEST_F(AtmosphericDragTest, DragOpposesVelocity) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    // Test various velocity directions
    std::vector<Eigen::Vector3d> velocities = {
        Eigen::Vector3d(7500.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 7500.0),
        Eigen::Vector3d(5000.0, 5000.0, 0.0),
        Eigen::Vector3d(1000.0, 2000.0, 3000.0)
    };
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);  // 400 km altitude
    
    for (const auto& vel : velocities) {
        ForceContext ctx = createContext(pos, vel);
        Eigen::Vector3d accel = drag.compute_acceleration(ctx);
        
        // Drag should oppose velocity
        Eigen::Vector3d vel_unit = vel.normalized();
        Eigen::Vector3d accel_unit = accel.normalized();
        
        // Dot product should be negative (opposite directions)
        EXPECT_LT(vel_unit.dot(accel_unit), -0.999);
    }
}

TEST_F(AtmosphericDragTest, DragScalesWithVelocitySquared) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);  // 400 km altitude
    
    Eigen::Vector3d vel1(0.0, 7500.0, 0.0);
    Eigen::Vector3d vel2(0.0, 15000.0, 0.0);  // 2x velocity
    
    ForceContext ctx1 = createContext(pos, vel1);
    ForceContext ctx2 = createContext(pos, vel2);
    
    Eigen::Vector3d accel1 = drag.compute_acceleration(ctx1);
    Eigen::Vector3d accel2 = drag.compute_acceleration(ctx2);
    
    // Drag should scale with v², so accel2 should be ~4x accel1
    double ratio = accel2.norm() / accel1.norm();
    EXPECT_NEAR(ratio, 4.0, 0.01);
}

TEST_F(AtmosphericDragTest, DragDecreasesWithAltitude) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    // Test at different altitudes
    Eigen::Vector3d pos_200km(6.578e6, 0.0, 0.0);  // 200 km
    Eigen::Vector3d pos_400km(6.778e6, 0.0, 0.0);  // 400 km
    Eigen::Vector3d pos_600km(6.978e6, 0.0, 0.0);  // 600 km
    
    ForceContext ctx_200 = createContext(pos_200km, vel);
    ForceContext ctx_400 = createContext(pos_400km, vel);
    ForceContext ctx_600 = createContext(pos_600km, vel);
    
    Eigen::Vector3d accel_200 = drag.compute_acceleration(ctx_200);
    Eigen::Vector3d accel_400 = drag.compute_acceleration(ctx_400);
    Eigen::Vector3d accel_600 = drag.compute_acceleration(ctx_600);
    
    // Higher altitude should have less drag (lower density)
    EXPECT_GT(accel_200.norm(), accel_400.norm());
    EXPECT_GT(accel_400.norm(), accel_600.norm());
}

TEST_F(AtmosphericDragTest, DragAtVeryHighAltitude) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    // At very high altitude (1000 km), drag should be extremely small
    Eigen::Vector3d pos(7.378e6, 0.0, 0.0);  // 1000 km altitude
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    Eigen::Vector3d accel = drag.compute_acceleration(ctx);
    
    // Should be very small but not exactly zero
    EXPECT_LT(accel.norm(), 1e-6);
}

TEST_F(AtmosphericDragTest, DragScalesWithMass) {
    double mass1 = 100.0;
    double mass2 = 200.0;
    
    AtmosphericDrag drag1(mass1, drag_coeff_, ref_area_);
    AtmosphericDrag drag2(mass2, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    
    Eigen::Vector3d accel1 = drag1.compute_acceleration(ctx);
    Eigen::Vector3d accel2 = drag2.compute_acceleration(ctx);
    
    // Heavier object should have less acceleration (same force)
    // accel ∝ 1/m, so accel2 should be half of accel1
    EXPECT_NEAR(accel2.norm() / accel1.norm(), 0.5, 0.01);
}

TEST_F(AtmosphericDragTest, DragScalesWithArea) {
    double area1 = 1.0;
    double area2 = 2.0;
    
    AtmosphericDrag drag1(mass_, drag_coeff_, area1);
    AtmosphericDrag drag2(mass_, drag_coeff_, area2);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    
    Eigen::Vector3d accel1 = drag1.compute_acceleration(ctx);
    Eigen::Vector3d accel2 = drag2.compute_acceleration(ctx);
    
    // Larger area should have more drag
    // accel ∝ A, so accel2 should be double accel1
    EXPECT_NEAR(accel2.norm() / accel1.norm(), 2.0, 0.01);
}

TEST_F(AtmosphericDragTest, DragScalesWithDragCoefficient) {
    double cd1 = 1.5;
    double cd2 = 3.0;
    
    AtmosphericDrag drag1(mass_, cd1, ref_area_);
    AtmosphericDrag drag2(mass_, cd2, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    
    Eigen::Vector3d accel1 = drag1.compute_acceleration(ctx);
    Eigen::Vector3d accel2 = drag2.compute_acceleration(ctx);
    
    // accel ∝ Cd, so accel2 should be double accel1
    EXPECT_NEAR(accel2.norm() / accel1.norm(), 2.0, 0.01);
}

TEST_F(AtmosphericDragTest, TimeIndependent) {
    // Drag should be time-independent (autonomous)
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx1 = createContext(pos, vel, 0.0);
    ForceContext ctx2 = createContext(pos, vel, 1000.0);
    ForceContext ctx3 = createContext(pos, vel, 1e6);
    
    Eigen::Vector3d accel1 = drag.compute_acceleration(ctx1);
    Eigen::Vector3d accel2 = drag.compute_acceleration(ctx2);
    Eigen::Vector3d accel3 = drag.compute_acceleration(ctx3);
    
    EXPECT_TRUE(accel1.isApprox(accel2, 1e-15));
    EXPECT_TRUE(accel1.isApprox(accel3, 1e-15));
}

// ============================================================================
// JACOBIAN TESTS
// ============================================================================

TEST_F(AtmosphericDragTest, JacobianZeroVelocity) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    ForceContext ctx = createContext(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d::Zero()
    );
    
    auto [da_dr, da_dv] = drag.compute_jacobian(ctx);
    
    // Both Jacobians should be zero when velocity is zero
    EXPECT_TRUE(da_dr.isApprox(Eigen::Matrix3d::Zero(), 1e-15));
    EXPECT_TRUE(da_dv.isApprox(Eigen::Matrix3d::Zero(), 1e-15));
}

TEST_F(AtmosphericDragTest, JacobianFinite) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    ForceContext ctx = createContext(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    auto [da_dr, da_dv] = drag.compute_jacobian(ctx);
    
    EXPECT_TRUE(da_dr.allFinite());
    EXPECT_TRUE(da_dv.allFinite());
}

TEST_F(AtmosphericDragTest, JacobianPositionNumericalValidation) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    auto [da_dr_analytical, da_dv] = drag.compute_jacobian(ctx);
    
    // Numerical Jacobian with finite differences
    const double h = 1.0;  // 1 meter perturbation
    Eigen::Matrix3d da_dr_numerical = Eigen::Matrix3d::Zero();
    
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d pos_plus = pos;
        Eigen::Vector3d pos_minus = pos;
        pos_plus(i) += h;
        pos_minus(i) -= h;
        
        ForceContext ctx_plus = createContext(pos_plus, vel);
        ForceContext ctx_minus = createContext(pos_minus, vel);
        
        Eigen::Vector3d accel_plus = drag.compute_acceleration(ctx_plus);
        Eigen::Vector3d accel_minus = drag.compute_acceleration(ctx_minus);
        
        da_dr_numerical.col(i) = (accel_plus - accel_minus) / (2.0 * h);
    }
    
    // Should match within reasonable tolerance
    EXPECT_TRUE(da_dr_analytical.isApprox(da_dr_numerical, 1e-8));
}

TEST_F(AtmosphericDragTest, JacobianVelocityNumericalValidation) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    auto [da_dr, da_dv_analytical] = drag.compute_jacobian(ctx);
    
    // Numerical Jacobian with finite differences
    const double h = 1.0;  // 1 m/s perturbation
    Eigen::Matrix3d da_dv_numerical = Eigen::Matrix3d::Zero();
    
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d vel_plus = vel;
        Eigen::Vector3d vel_minus = vel;
        vel_plus(i) += h;
        vel_minus(i) -= h;
        
        ForceContext ctx_plus = createContext(pos, vel_plus);
        ForceContext ctx_minus = createContext(pos, vel_minus);
        
        Eigen::Vector3d accel_plus = drag.compute_acceleration(ctx_plus);
        Eigen::Vector3d accel_minus = drag.compute_acceleration(ctx_minus);
        
        da_dv_numerical.col(i) = (accel_plus - accel_minus) / (2.0 * h);
    }
    
    // Should match within reasonable tolerance
    EXPECT_TRUE(da_dv_analytical.isApprox(da_dv_numerical, 1e-6));
}

TEST_F(AtmosphericDragTest, JacobianVelocityArbitraryDirection) {
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(5000.0, 3000.0, 2000.0);  // Arbitrary direction
    
    ForceContext ctx = createContext(pos, vel);
    auto [da_dr, da_dv_analytical] = drag.compute_jacobian(ctx);
    
    // Numerical validation
    const double h = 1.0;
    Eigen::Matrix3d da_dv_numerical = Eigen::Matrix3d::Zero();
    
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d vel_plus = vel;
        Eigen::Vector3d vel_minus = vel;
        vel_plus(i) += h;
        vel_minus(i) -= h;
        
        ForceContext ctx_plus = createContext(pos, vel_plus);
        ForceContext ctx_minus = createContext(pos, vel_minus);
        
        Eigen::Vector3d accel_plus = drag.compute_acceleration(ctx_plus);
        Eigen::Vector3d accel_minus = drag.compute_acceleration(ctx_minus);
        
        da_dv_numerical.col(i) = (accel_plus - accel_minus) / (2.0 * h);
    }
    
    EXPECT_TRUE(da_dv_analytical.isApprox(da_dv_numerical, 1e-6));
}

TEST_F(AtmosphericDragTest, JacobianSymmetry) {
    // Velocity Jacobian should be symmetric for drag force
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    auto [da_dr, da_dv] = drag.compute_jacobian(ctx);
    
    // da_dv should be symmetric
    EXPECT_TRUE(da_dv.isApprox(da_dv.transpose(), 1e-10));
}

// ============================================================================
// INTEGRATION WITH POINT MASS DYNAMICS
// ============================================================================

TEST_F(AtmosphericDragTest, IntegrationPointMassDynamics) {
    // Create drag force
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    // Create point mass dynamics with drag
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(drag);
    PointMassDynamics dynamics(forces);
    
    // State at 400 km altitude
    Eigen::VectorXd state(6);
    state << 6.771e6, 0.0, 0.0,  // Position (m)
             0.0, 7500.0, 0.0;    // Velocity (m/s)
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // Check dimensions
    EXPECT_EQ(state_dot.size(), 6);
    
    // Velocity derivative (position rate)
    EXPECT_DOUBLE_EQ(state_dot(0), 0.0);
    EXPECT_DOUBLE_EQ(state_dot(1), 7500.0);
    EXPECT_DOUBLE_EQ(state_dot(2), 0.0);
    
    // Acceleration should oppose velocity
    Eigen::Vector3d accel = state_dot.segment<3>(3);
    Eigen::Vector3d vel = state.segment<3>(3);
    
    EXPECT_LT(accel.normalized().dot(vel.normalized()), -0.999);
    EXPECT_GT(accel.norm(), 0.0);
}

TEST_F(AtmosphericDragTest, IntegrationPointMassWithGravity) {
    // Create forces: gravity + drag
    auto gravity = std::make_shared<PointMassGravity>();
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(gravity);
    forces.push_back(drag);
    
    PointMassDynamics dynamics(forces);
    
    // LEO orbit state
    Eigen::VectorXd state(6);
    state << 6.771e6, 0.0, 0.0,
             0.0, 7500.0, 0.0;
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // Total acceleration should be gravity + drag
    Eigen::Vector3d total_accel = state_dot.segment<3>(3);
    
    // Gravity alone
    ForceContext ctx_gravity;
    ctx_gravity.position = state.segment<3>(0);
    ctx_gravity.velocity = state.segment<3>(3);
    Eigen::Vector3d gravity_accel = gravity->compute_acceleration(ctx_gravity);
    
    // Drag alone
    Eigen::Vector3d drag_accel = drag->compute_acceleration(ctx_gravity);
    
    // Should sum correctly
    EXPECT_TRUE(total_accel.isApprox(gravity_accel + drag_accel, 1e-12));
}

TEST_F(AtmosphericDragTest, IntegrationPointMassJacobian) {
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(drag);
    PointMassDynamics dynamics(forces);
    
    Eigen::VectorXd state(6);
    state << 6.771e6, 0.0, 0.0,
             0.0, 7500.0, 0.0;
    
    Eigen::MatrixXd jacobian = dynamics.compute_jacobian(0.0, state);
    
    // Check structure: [  0    I  ]
    //                  [da/dr da/dv]
    EXPECT_EQ(jacobian.rows(), 6);
    EXPECT_EQ(jacobian.cols(), 6);
    
    // Upper-left should be zero
    EXPECT_TRUE((jacobian.block<3, 3>(0, 0).isApprox(Eigen::Matrix3d::Zero(), 1e-15)));
    
    // Upper-right should be identity
    EXPECT_TRUE((jacobian.block<3, 3>(0, 3).isApprox(Eigen::Matrix3d::Identity(), 1e-15)));
    
    // Lower blocks should match drag Jacobians
    ForceContext ctx;
    ctx.position = state.segment<3>(0);
    ctx.velocity = state.segment<3>(3);
    auto [da_dr, da_dv] = drag->compute_jacobian(ctx);
    
    EXPECT_TRUE((jacobian.block<3, 3>(3, 0).isApprox(da_dr, 1e-12)));
    EXPECT_TRUE((jacobian.block<3, 3>(3, 3).isApprox(da_dv, 1e-12)));
}

TEST_F(AtmosphericDragTest, IntegrationPointMassOrbitDecay) {
    // Simulate short orbit segment to verify drag causes decay
    auto gravity = std::make_shared<PointMassGravity>();
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    std::vector<std::shared_ptr<IForce>> forces_with_drag;
    forces_with_drag.push_back(gravity);
    forces_with_drag.push_back(drag);
    
    std::vector<std::shared_ptr<IForce>> forces_no_drag;
    forces_no_drag.push_back(gravity);
    
    PointMassDynamics dynamics_with_drag(forces_with_drag);
    PointMassDynamics dynamics_no_drag(forces_no_drag);
    
    // Initial state
    Eigen::VectorXd state(6);
    state << 6.771e6, 0.0, 0.0,
             0.0, 7500.0, 0.0;
    
    Eigen::VectorXd state_dot_with = dynamics_with_drag.compute_dynamics(0.0, state);
    Eigen::VectorXd state_dot_no = dynamics_no_drag.compute_dynamics(0.0, state);
    
    // With drag, acceleration magnitude should be greater
    // (gravity pulls inward, drag opposes motion)
    Eigen::Vector3d accel_with = state_dot_with.segment<3>(3);
    Eigen::Vector3d accel_no = state_dot_no.segment<3>(3);
    
    // Different accelerations
    EXPECT_GT((accel_with - accel_no).norm(), 1e-6);
}

// ============================================================================
// INTEGRATION WITH RIGID BODY DYNAMICS
// ============================================================================

TEST_F(AtmosphericDragTest, IntegrationRigidBodyDynamics) {
    // Create drag force
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(drag);
    
    std::vector<std::shared_ptr<ITorque>> torques;  // No torques for this test
    
    // Inertia tensor
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity() * 10.0;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia, mass_);
    
    // State: [r(3), v(3), q(4), ω(3)]
    Eigen::VectorXd state(13);
    state << 6.771e6, 0.0, 0.0,      // Position
             0.0, 7500.0, 0.0,        // Velocity
             0.0, 0.0, 0.0, 1.0,      // Quaternion (identity)
             0.0, 0.0, 0.0;           // Angular velocity
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_EQ(state_dot.size(), 13);
    
    // Check translational dynamics
    EXPECT_TRUE(state_dot.segment<3>(0).isApprox(state.segment<3>(3), 1e-15));  // dr/dt = v
    
    // Acceleration should oppose velocity
    Eigen::Vector3d accel = state_dot.segment<3>(3);
    Eigen::Vector3d vel = state.segment<3>(3);
    EXPECT_LT(accel.normalized().dot(vel.normalized()), -0.999);
}

TEST_F(AtmosphericDragTest, IntegrationRigidBodyWithGravity) {
    auto gravity = std::make_shared<PointMassGravity>();
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(gravity);
    forces.push_back(drag);
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity() * 10.0;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia, mass_);
    
    Eigen::VectorXd state(13);
    state << 6.771e6, 0.0, 0.0,
             0.0, 7500.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
             0.0, 0.0, 0.0;
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // Should have both gravity and drag accelerations
    Eigen::Vector3d total_accel = state_dot.segment<3>(3);
    
    // Verify non-zero acceleration
    EXPECT_GT(total_accel.norm(), 8.0);  // Should be dominated by gravity (~8.7 m/s²)
}

TEST_F(AtmosphericDragTest, IntegrationRigidBodyRotatingFrame) {
    // Test with non-zero angular velocity
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(drag);
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity() * 10.0;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia, mass_);
    
    Eigen::VectorXd state(13);
    state << 6.771e6, 0.0, 0.0,
             0.0, 7500.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
             0.1, 0.05, 0.08;  // Non-zero angular velocity
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // Translational dynamics should still work
    EXPECT_TRUE(state_dot.segment<3>(0).isApprox(state.segment<3>(3), 1e-15));
    
    // Should have drag acceleration
    Eigen::Vector3d accel = state_dot.segment<3>(3);
    EXPECT_GT(accel.norm(), 0.0);
    EXPECT_TRUE(accel.allFinite());
}

TEST_F(AtmosphericDragTest, IntegrationRigidBodyJacobian) {
    auto drag = std::make_shared<AtmosphericDrag>(mass_, drag_coeff_, ref_area_);
    
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(drag);
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity() * 10.0;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia, mass_);
    
    Eigen::VectorXd state(13);
    state << 6.771e6, 0.0, 0.0,
             0.0, 7500.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
             0.0, 0.0, 0.0;
    
    Eigen::MatrixXd jacobian = dynamics.compute_jacobian(0.0, state);
    
    // Should be 13x13
    EXPECT_EQ(jacobian.rows(), 13);
    EXPECT_EQ(jacobian.cols(), 13);
    
    EXPECT_TRUE(jacobian.allFinite());
}

// ============================================================================
// PHYSICAL VALIDATION TESTS
// ============================================================================

TEST_F(AtmosphericDragTest, RealisticDragMagnitude) {
    // Verify drag produces reasonable acceleration at LEO
    AtmosphericDrag drag(mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);  // 400 km
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);   // ~LEO orbital velocity
    
    ForceContext ctx = createContext(pos, vel);
    Eigen::Vector3d accel = drag.compute_acceleration(ctx);
    
    // At 400 km, drag should be on order of 1e-5 to 1e-6 m/s² for typical satellite
    EXPECT_GT(accel.norm(), 1e-7);
    EXPECT_LT(accel.norm(), 1e-3);
}

TEST_F(AtmosphericDragTest, BallisticCoefficient) {
    // Higher ballistic coefficient (β = m/(Cd*A)) means less drag
    double beta1 = mass_ / (drag_coeff_ * ref_area_);  // β = 100 / 2.2 ≈ 45.5 kg/m²
    double beta2 = 2.0 * beta1;  // Double the ballistic coefficient
    
    // Achieve beta2 by doubling mass
    AtmosphericDrag drag1(mass_, drag_coeff_, ref_area_);
    AtmosphericDrag drag2(2.0 * mass_, drag_coeff_, ref_area_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    
    ForceContext ctx = createContext(pos, vel);
    
    Eigen::Vector3d accel1 = drag1.compute_acceleration(ctx);
    Eigen::Vector3d accel2 = drag2.compute_acceleration(ctx);
    
    // Higher β should have half the acceleration
    EXPECT_NEAR(accel2.norm() / accel1.norm(), 0.5, 0.01);
}
