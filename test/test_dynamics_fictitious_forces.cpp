#include <gtest/gtest.h>

#include "dynamics/fictitious_forces.hpp"
#include "dynamics/force.hpp"

#include <Eigen/Dense>

#include <cmath>

using namespace dynamics;

// Test fixture
class FictitiousForcesTest : public ::testing::Test {
protected:
    // Earth's angular velocity (rad/s)
    static constexpr double EARTH_OMEGA = 7.292115e-5;
    
    Eigen::Vector3d earth_omega_;
    
    void SetUp() override {
        earth_omega_ << 0.0, 0.0, EARTH_OMEGA;
    }
};

// Test zero velocity gives zero Coriolis force
TEST_F(FictitiousForcesTest, ZeroVelocityZeroCoriolis) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 6.378e6, 0.0, 0.0; // On equator
    ctx.velocity << 0.0, 0.0, 0.0;     // Zero velocity
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Coriolis should be zero, only centrifugal remains
    // Centrifugal: -omega x (omega x r)
    Eigen::Vector3d centrifugal = -earth_omega_.cross(earth_omega_.cross(ctx.position));
    
    EXPECT_NEAR(acceleration.norm(), centrifugal.norm(), 1e-10);
}

// Test position at origin gives zero centrifugal force
TEST_F(FictitiousForcesTest, OriginZeroCentrifugal) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 0.0, 0.0, 0.0;     // At origin
    ctx.velocity << 100.0, 100.0, 0.0; // Some velocity
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Centrifugal should be zero, only Coriolis remains
    // Coriolis: -2 * omega x v
    Eigen::Vector3d coriolis = -2.0 * earth_omega_.cross(ctx.velocity);
    
    EXPECT_NEAR(acceleration(0), coriolis(0), 1e-10);
    EXPECT_NEAR(acceleration(1), coriolis(1), 1e-10);
    EXPECT_NEAR(acceleration(2), coriolis(2), 1e-10);
}

// Test Coriolis force calculation
TEST_F(FictitiousForcesTest, CoriolisForce) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 0.0, 0.0, 0.0;     // At origin to isolate Coriolis
    ctx.velocity << 100.0, 0.0, 0.0;   // Eastward velocity
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Coriolis: -2 * omega x v
    // omega = [0, 0, w], v = [100, 0, 0]
    // omega x v = [0, 100*w, 0]
    // -2 * omega x v = [0, -200*w, 0]
    
    EXPECT_NEAR(acceleration(0), 0.0, 1e-15);
    EXPECT_NEAR(acceleration(1), -200.0 * EARTH_OMEGA, 1e-10);  // Negative!
    EXPECT_NEAR(acceleration(2), 0.0, 1e-15);
}

// Test centrifugal force calculation
TEST_F(FictitiousForcesTest, CentrifugalForce) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 6.378e6, 0.0, 0.0; // On equator, 1 Earth radius
    ctx.velocity << 0.0, 0.0, 0.0;     // Zero velocity to isolate centrifugal
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Centrifugal: -omega x (omega x r)
    // omega = [0, 0, w], r = [r, 0, 0]
    // omega x r = [0, r*w, 0]
    // omega x (omega x r) = [-r*w^2, 0, 0]
    // -omega x (omega x r) = [r*w^2, 0, 0]
    
    double expected = 6.378e6 * EARTH_OMEGA * EARTH_OMEGA;
    EXPECT_NEAR(acceleration(0), expected, 1e-6);
    EXPECT_NEAR(acceleration(1), 0.0, 1e-10);
    EXPECT_NEAR(acceleration(2), 0.0, 1e-10);
}

// Test combined Coriolis and centrifugal forces
TEST_F(FictitiousForcesTest, CombinedForces) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 6.378e6, 0.0, 0.0;
    ctx.velocity << 0.0, 100.0, 0.0;   // Northward velocity
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Both forces should be present
    Eigen::Vector3d coriolis = -2.0 * earth_omega_.cross(ctx.velocity);
    Eigen::Vector3d centrifugal = -earth_omega_.cross(earth_omega_.cross(ctx.position));
    Eigen::Vector3d expected = coriolis + centrifugal;
    
    EXPECT_NEAR(acceleration(0), expected(0), 1e-6);
    EXPECT_NEAR(acceleration(1), expected(1), 1e-10);
    EXPECT_NEAR(acceleration(2), expected(2), 1e-10);
}

// Test Coriolis perpendicular to velocity
TEST_F(FictitiousForcesTest, CoriolisPerpendicularToVelocity) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 0.0, 0.0, 0.0;
    ctx.velocity << 100.0, 50.0, 25.0;
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Coriolis force should be perpendicular to velocity
    double dot_product = acceleration.dot(ctx.velocity);
    EXPECT_NEAR(dot_product, 0.0, 1e-6);
}

// Test with custom angular velocity
TEST_F(FictitiousForcesTest, CustomAngularVelocity) {
    Eigen::Vector3d custom_omega(1.0, 2.0, 3.0);
    FictitiousForces forces(custom_omega);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1.0, 2.0, 3.0;
    ctx.velocity << 4.0, 5.0, 6.0;
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Calculate expected
    Eigen::Vector3d coriolis = -2.0 * custom_omega.cross(ctx.velocity);
    Eigen::Vector3d centrifugal = -custom_omega.cross(custom_omega.cross(ctx.position));
    Eigen::Vector3d expected = coriolis + centrifugal;
    
    EXPECT_NEAR(acceleration(0), expected(0), 1e-10);
    EXPECT_NEAR(acceleration(1), expected(1), 1e-10);
    EXPECT_NEAR(acceleration(2), expected(2), 1e-10);
}

// Test with zero angular velocity (no fictitious forces)
TEST_F(FictitiousForcesTest, ZeroAngularVelocity) {
    Eigen::Vector3d zero_omega = Eigen::Vector3d::Zero();
    FictitiousForces forces(zero_omega);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1000.0, 2000.0, 3000.0;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // No rotation = no fictitious forces
    EXPECT_DOUBLE_EQ(acceleration(0), 0.0);
    EXPECT_DOUBLE_EQ(acceleration(1), 0.0);
    EXPECT_DOUBLE_EQ(acceleration(2), 0.0);
}

// Test Coriolis magnitude scales with velocity
TEST_F(FictitiousForcesTest, CoriolisScalesWithVelocity) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 0.0, 0.0, 0.0;
    
    // Test with velocity v
    ctx.velocity << 100.0, 0.0, 0.0;
    Eigen::Vector3d acc1 = forces.compute_force(ctx);
    
    // Test with velocity 2v
    ctx.velocity << 200.0, 0.0, 0.0;
    Eigen::Vector3d acc2 = forces.compute_force(ctx);
    
    // Acceleration should double
    EXPECT_NEAR(acc2.norm(), 2.0 * acc1.norm(), 1e-10);
}

// Test centrifugal magnitude scales with position
TEST_F(FictitiousForcesTest, CentrifugalScalesWithPosition) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.velocity << 0.0, 0.0, 0.0;
    
    // Test at radius r
    ctx.position << 1e6, 0.0, 0.0;
    Eigen::Vector3d acc1 = forces.compute_force(ctx);
    
    // Test at radius 2r
    ctx.position << 2e6, 0.0, 0.0;
    Eigen::Vector3d acc2 = forces.compute_force(ctx);
    
    // Acceleration should double
    EXPECT_NEAR(acc2.norm(), 2.0 * acc1.norm(), 1e-10);
}

// Test Coriolis direction reverses with velocity
TEST_F(FictitiousForcesTest, CoriolisDirectionReversal) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 0.0, 0.0, 0.0;
    
    // Test with +velocity
    ctx.velocity << 100.0, 0.0, 0.0;
    Eigen::Vector3d acc_pos = forces.compute_force(ctx);
    
    // Test with -velocity
    ctx.velocity << -100.0, 0.0, 0.0;
    Eigen::Vector3d acc_neg = forces.compute_force(ctx);
    
    // Accelerations should be opposite
    EXPECT_NEAR(acc_pos(0), -acc_neg(0), 1e-10);
    EXPECT_NEAR(acc_pos(1), -acc_neg(1), 1e-10);
    EXPECT_NEAR(acc_pos(2), -acc_neg(2), 1e-10);
}

// Test centrifugal always points outward
TEST_F(FictitiousForcesTest, CentrifugalOutward) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.velocity << 0.0, 0.0, 0.0;
    
    // Test at different positions on equatorial plane
    std::vector<Eigen::Vector3d> positions = {
        Eigen::Vector3d(1e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 1e6, 0.0),
        Eigen::Vector3d(-1e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, -1e6, 0.0)
    };
    
    for (const auto& pos : positions) {
        ctx.position = pos;
        Eigen::Vector3d acc = forces.compute_force(ctx);
        
        // Centrifugal should point away from rotation axis
        // For z-axis rotation, this is in xy-plane
        Eigen::Vector3d radial = pos - Eigen::Vector3d(0.0, 0.0, pos(2));
        if (radial.norm() > 1e-10) {
            radial.normalize();
            Eigen::Vector3d acc_xy(acc(0), acc(1), 0.0);
            acc_xy.normalize();
            
            double dot = radial.dot(acc_xy);
            EXPECT_GT(dot, 0.99); // Should point outward
        }
    }
}

// Test time independence
TEST_F(FictitiousForcesTest, TimeIndependent) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.position << 6.378e6, 0.0, 0.0;
    ctx.velocity << 0.0, 100.0, 0.0;
    
    // Test at different times
    ctx.t = 0.0;
    Eigen::Vector3d acc1 = forces.compute_force(ctx);
    
    ctx.t = 100.0;
    Eigen::Vector3d acc2 = forces.compute_force(ctx);
    
    ctx.t = -50.0;
    Eigen::Vector3d acc3 = forces.compute_force(ctx);
    
    // Fictitious forces don't depend on time
    EXPECT_EQ(acc1, acc2);
    EXPECT_EQ(acc1, acc3);
}

// Test typical Earth surface values
TEST_F(FictitiousForcesTest, EarthSurfaceTypicalValues) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 6.378e6, 0.0, 0.0; // Earth radius
    ctx.velocity << 0.0, 100.0, 0.0;   // 100 m/s northward
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Magnitude should be small compared to gravity (~0.03 m/s^2)
    EXPECT_LT(acceleration.norm(), 1.0);
    EXPECT_GT(acceleration.norm(), 1e-6);
}

// Test high angular velocity (rapidly rotating frame)
TEST_F(FictitiousForcesTest, HighAngularVelocity) {
    Eigen::Vector3d fast_omega(0.0, 0.0, 1.0); // 1 rad/s
    FictitiousForces forces(fast_omega);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1.0, 0.0, 0.0;
    ctx.velocity << 0.0, 1.0, 0.0;
    
    Eigen::Vector3d acceleration = forces.compute_force(ctx);
    
    // Forces should be significant
    EXPECT_GT(acceleration.norm(), 1.0);
}
