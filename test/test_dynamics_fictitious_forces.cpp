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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    Eigen::Vector3d acc1 = forces.compute_acceleration(ctx);
    
    // Test with velocity 2v
    ctx.velocity << 200.0, 0.0, 0.0;
    Eigen::Vector3d acc2 = forces.compute_acceleration(ctx);
    
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
    Eigen::Vector3d acc1 = forces.compute_acceleration(ctx);
    
    // Test at radius 2r
    ctx.position << 2e6, 0.0, 0.0;
    Eigen::Vector3d acc2 = forces.compute_acceleration(ctx);
    
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
    Eigen::Vector3d acc_pos = forces.compute_acceleration(ctx);
    
    // Test with -velocity
    ctx.velocity << -100.0, 0.0, 0.0;
    Eigen::Vector3d acc_neg = forces.compute_acceleration(ctx);
    
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
        Eigen::Vector3d acc = forces.compute_acceleration(ctx);
        
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
    Eigen::Vector3d acc1 = forces.compute_acceleration(ctx);
    
    ctx.t = 100.0;
    Eigen::Vector3d acc2 = forces.compute_acceleration(ctx);
    
    ctx.t = -50.0;
    Eigen::Vector3d acc3 = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
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
    
    Eigen::Vector3d acceleration = forces.compute_acceleration(ctx);
    
    // Forces should be significant
    EXPECT_GT(acceleration.norm(), 1.0);
}

// Add these tests after the existing tests:

// ============================================================================
// JACOBIAN TESTS
// ============================================================================

TEST_F(FictitiousForcesTest, JacobianPositionDerivative) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 6.378e6, 1e6, 2e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    // Get analytical Jacobian
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // Numerical Jacobian for position
    double epsilon = 1e-4;  // Larger epsilon for finite differences
    Eigen::Matrix3d da_dr_numerical;
    
    for (int i = 0; i < 3; ++i) {
        ForceContext ctx_plus = ctx;
        ForceContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d a_plus = forces.compute_acceleration(ctx_plus);
        Eigen::Vector3d a_minus = forces.compute_acceleration(ctx_minus);
        
        da_dr_numerical.col(i) = (a_plus - a_minus) / (2.0 * epsilon);
    }
    
    // Compare
    double error = (da_dr - da_dr_numerical).norm();
    EXPECT_LT(error, 1e-6) << "Position Jacobian error: " << error;
}

TEST_F(FictitiousForcesTest, JacobianVelocityDerivative) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 6.378e6, 1e6, 2e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    // Get analytical Jacobian
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // Numerical Jacobian for velocity
    double epsilon = 1e-4;
    Eigen::Matrix3d da_dv_numerical;
    
    for (int i = 0; i < 3; ++i) {
        ForceContext ctx_plus = ctx;
        ForceContext ctx_minus = ctx;
        
        ctx_plus.velocity(i) += epsilon;
        ctx_minus.velocity(i) -= epsilon;
        
        Eigen::Vector3d a_plus = forces.compute_acceleration(ctx_plus);
        Eigen::Vector3d a_minus = forces.compute_acceleration(ctx_minus);
        
        da_dv_numerical.col(i) = (a_plus - a_minus) / (2.0 * epsilon);
    }
    
    // Compare
    double error = (da_dv - da_dv_numerical).norm();
    EXPECT_LT(error, 1e-6) << "Velocity Jacobian error: " << error;
}

TEST_F(FictitiousForcesTest, JacobianPositionStructure) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // ∂a/∂r = -ω×(ω×I) = -[ω×]²
    // For ω = [0, 0, w]:
    // [ω×] = [0  -w  0]
    //        [w   0  0]
    //        [0   0  0]
    //
    // [ω×]² = [-w²  0  0]
    //         [ 0 -w²  0]
    //         [ 0   0  0]
    //
    // So ∂a/∂r = [w²  0  0]
    //            [ 0 w²  0]
    //            [ 0  0  0]
    
    double w = EARTH_OMEGA;
    double w2 = w * w;
    
    EXPECT_NEAR(da_dr(0, 0), w2, 1e-15);
    EXPECT_NEAR(da_dr(1, 1), w2, 1e-15);
    EXPECT_NEAR(da_dr(2, 2), 0.0, 1e-15);
    
    // Off-diagonal should be zero for z-axis rotation
    EXPECT_NEAR(da_dr(0, 1), 0.0, 1e-15);
    EXPECT_NEAR(da_dr(0, 2), 0.0, 1e-15);
    EXPECT_NEAR(da_dr(1, 0), 0.0, 1e-15);
    EXPECT_NEAR(da_dr(1, 2), 0.0, 1e-15);
    EXPECT_NEAR(da_dr(2, 0), 0.0, 1e-15);
    EXPECT_NEAR(da_dr(2, 1), 0.0, 1e-15);
}

TEST_F(FictitiousForcesTest, JacobianVelocityStructure) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // ∂a/∂v = -2[ω×]
    // For ω = [0, 0, w]:
    // -2[ω×] = [ 0  2w  0]
    //          [-2w  0  0]
    //          [ 0   0  0]
    
    double w = EARTH_OMEGA;
    
    EXPECT_NEAR(da_dv(0, 0), 0.0, 1e-15);
    EXPECT_NEAR(da_dv(0, 1), 2.0 * w, 1e-15);
    EXPECT_NEAR(da_dv(0, 2), 0.0, 1e-15);
    
    EXPECT_NEAR(da_dv(1, 0), -2.0 * w, 1e-15);
    EXPECT_NEAR(da_dv(1, 1), 0.0, 1e-15);
    EXPECT_NEAR(da_dv(1, 2), 0.0, 1e-15);
    
    EXPECT_NEAR(da_dv(2, 0), 0.0, 1e-15);
    EXPECT_NEAR(da_dv(2, 1), 0.0, 1e-15);
    EXPECT_NEAR(da_dv(2, 2), 0.0, 1e-15);
}

TEST_F(FictitiousForcesTest, JacobianSkewSymmetricVelocity) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // ∂a/∂v = -2[ω×] should be skew-symmetric
    Eigen::Matrix3d da_dv_transpose = da_dv.transpose();
    
    EXPECT_NEAR((da_dv + da_dv_transpose).norm(), 0.0, 1e-14)
        << "Velocity Jacobian should be skew-symmetric";
}

TEST_F(FictitiousForcesTest, JacobianSymmetricPosition) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // ∂a/∂r = -[ω×]² should be symmetric
    Eigen::Matrix3d da_dr_transpose = da_dr.transpose();
    
    EXPECT_NEAR((da_dr - da_dr_transpose).norm(), 0.0, 1e-14)
        << "Position Jacobian should be symmetric";
}

TEST_F(FictitiousForcesTest, JacobianCustomOmega) {
    Eigen::Vector3d custom_omega(1.0, 2.0, 3.0);
    FictitiousForces forces(custom_omega);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 100.0, 200.0, 300.0;
    ctx.velocity << 10.0, 20.0, 30.0;
    
    // Get analytical Jacobian
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // Numerical verification
    double epsilon = 1e-6;
    
    // Check position Jacobian
    Eigen::Matrix3d da_dr_numerical;
    for (int i = 0; i < 3; ++i) {
        ForceContext ctx_plus = ctx;
        ForceContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d a_plus = forces.compute_acceleration(ctx_plus);
        Eigen::Vector3d a_minus = forces.compute_acceleration(ctx_minus);
        
        da_dr_numerical.col(i) = (a_plus - a_minus) / (2.0 * epsilon);
    }
    
    EXPECT_LT((da_dr - da_dr_numerical).norm(), 1e-5);
    
    // Check velocity Jacobian
    Eigen::Matrix3d da_dv_numerical;
    for (int i = 0; i < 3; ++i) {
        ForceContext ctx_plus = ctx;
        ForceContext ctx_minus = ctx;
        
        ctx_plus.velocity(i) += epsilon;
        ctx_minus.velocity(i) -= epsilon;
        
        Eigen::Vector3d a_plus = forces.compute_acceleration(ctx_plus);
        Eigen::Vector3d a_minus = forces.compute_acceleration(ctx_minus);
        
        da_dv_numerical.col(i) = (a_plus - a_minus) / (2.0 * epsilon);
    }
    
    EXPECT_LT((da_dv - da_dv_numerical).norm(), 1e-5);
}

TEST_F(FictitiousForcesTest, JacobianZeroOmega) {
    Eigen::Vector3d zero_omega = Eigen::Vector3d::Zero();
    FictitiousForces forces(zero_omega);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // With no rotation, all Jacobians should be zero
    EXPECT_DOUBLE_EQ(da_dr.norm(), 0.0);
    EXPECT_DOUBLE_EQ(da_dv.norm(), 0.0);
}

TEST_F(FictitiousForcesTest, JacobianStateIndependent) {
    FictitiousForces forces(earth_omega_);
    
    // Test that Jacobians don't depend on state values (linear transformation)
    ForceContext ctx1, ctx2;
    ctx1.t = 0.0;
    ctx1.position << 1e6, 2e6, 3e6;
    ctx1.velocity << 100.0, 200.0, 300.0;
    
    ctx2.t = 0.0;
    ctx2.position << 5e6, -1e6, 0.5e6;
    ctx2.velocity << -50.0, 100.0, 25.0;
    
    auto [da_dr1, da_dv1] = forces.compute_jacobian(ctx1);
    auto [da_dr2, da_dv2] = forces.compute_jacobian(ctx2);
    
    // Jacobians should be identical (linear transformations)
    EXPECT_EQ(da_dr1, da_dr2);
    EXPECT_EQ(da_dv1, da_dv2);
}

TEST_F(FictitiousForcesTest, JacobianLinearConsistency) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // Test linearity: a(r + dr, v + dv) ≈ a(r,v) + da_dr·dr + da_dv·dv
    Eigen::Vector3d dr(100.0, -50.0, 25.0);
    Eigen::Vector3d dv(1.0, 2.0, -0.5);
    
    ForceContext ctx_perturbed = ctx;
    ctx_perturbed.position += dr;
    ctx_perturbed.velocity += dv;
    
    Eigen::Vector3d a_base = forces.compute_acceleration(ctx);
    Eigen::Vector3d a_perturbed = forces.compute_acceleration(ctx_perturbed);
    
    Eigen::Vector3d a_linear = a_base + da_dr * dr + da_dv * dv;
    
    // For linear transformations, should be exact
    EXPECT_LT((a_perturbed - a_linear).norm(), 1e-10);
}

TEST_F(FictitiousForcesTest, JacobianPositionNegativeSemidefinite) {
    FictitiousForces forces(earth_omega_);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // ∂a/∂r = -[ω×]² should be negative semi-definite
    // (centrifugal force increases with distance from axis)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(da_dr);
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    
    // All eigenvalues should be >= 0 (force points outward)
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(eigenvalues(i), -1e-14) 
            << "Eigenvalue " << i << " is negative: " << eigenvalues(i);
    }
}

TEST_F(FictitiousForcesTest, JacobianHighAngularVelocity) {
    Eigen::Vector3d fast_omega(0.0, 0.0, 10.0); // 10 rad/s
    FictitiousForces forces(fast_omega);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 10.0, 20.0, 30.0;
    ctx.velocity << 1.0, 2.0, 3.0;
    
    // Get analytical Jacobian
    auto [da_dr, da_dv] = forces.compute_jacobian(ctx);
    
    // Numerical verification with high omega
    double epsilon = 1e-6;
    Eigen::Matrix3d da_dr_numerical;
    
    for (int i = 0; i < 3; ++i) {
        ForceContext ctx_plus = ctx;
        ForceContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d a_plus = forces.compute_acceleration(ctx_plus);
        Eigen::Vector3d a_minus = forces.compute_acceleration(ctx_minus);
        
        da_dr_numerical.col(i) = (a_plus - a_minus) / (2.0 * epsilon);
    }
    
    EXPECT_LT((da_dr - da_dr_numerical).norm(), 1e-4);
}

TEST_F(FictitiousForcesTest, JacobianMagnitudeScaling) {
    // Test that Jacobian magnitudes scale correctly with omega
    Eigen::Vector3d omega1(0.0, 0.0, 1.0);
    Eigen::Vector3d omega2(0.0, 0.0, 2.0);
    
    FictitiousForces forces1(omega1);
    FictitiousForces forces2(omega2);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 100.0, 200.0, 300.0;
    ctx.velocity << 10.0, 20.0, 30.0;
    
    auto [da_dr1, da_dv1] = forces1.compute_jacobian(ctx);
    auto [da_dr2, da_dv2] = forces2.compute_jacobian(ctx);
    
    // Position Jacobian should scale with omega²
    EXPECT_NEAR(da_dr2.norm(), 4.0 * da_dr1.norm(), 1e-10);
    
    // Velocity Jacobian should scale linearly with omega
    EXPECT_NEAR(da_dv2.norm(), 2.0 * da_dv1.norm(), 1e-10);
}
