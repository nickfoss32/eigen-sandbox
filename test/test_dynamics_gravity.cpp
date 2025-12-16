#include <gtest/gtest.h>
#include "dynamics/gravity.hpp"
#include <cmath>

using namespace dynamics;

// Earth gravitational parameter
constexpr double EARTH_MU = 3.986004418e14;  // m³/s²
constexpr double EARTH_J2 = 1.08263e-3;
constexpr double EARTH_RADIUS = 6.378137e6;  // m

// ============================================================================
// POINT MASS GRAVITY TESTS
// ============================================================================

class PointMassGravityTest : public ::testing::Test {
protected:
    PointMassGravity gravity_{EARTH_MU};
};

TEST_F(PointMassGravityTest, RadialAcceleration) {
    // At radius r, acceleration magnitude should be μ/r²
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 0.0, 0.0;  // 7000 km on x-axis
    ctx.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a = gravity_.compute_acceleration(ctx);
    double r = ctx.position.norm();
    double expected_mag = EARTH_MU / (r * r);
    
    EXPECT_NEAR(a.norm(), expected_mag, 1e-6);
}

TEST_F(PointMassGravityTest, DirectionTowardCenter) {
    // Acceleration should point toward origin
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 1e6, 2e6, 3e6;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a = gravity_.compute_acceleration(ctx);
    Eigen::Vector3d r_hat = ctx.position.normalized();
    Eigen::Vector3d a_hat = a.normalized();
    
    // a and r should be anti-parallel
    EXPECT_NEAR((r_hat + a_hat).norm(), 0.0, 1e-10);
}

TEST_F(PointMassGravityTest, CircularOrbitVelocity) {
    // For circular orbit: v = sqrt(μ/r)
    double r = 7e6;  // 7000 km
    double v_circular = std::sqrt(EARTH_MU / r);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << r, 0.0, 0.0;
    ctx.velocity << 0.0, v_circular, 0.0;
    
    Eigen::Vector3d a = gravity_.compute_acceleration(ctx);
    
    // Centripetal acceleration = v²/r
    double expected = v_circular * v_circular / r;
    EXPECT_NEAR(a.norm(), expected, 1e-3);
}

TEST_F(PointMassGravityTest, InverseSquareLaw) {
    // At 2r, acceleration should be 1/4 of acceleration at r
    ForceContext ctx1, ctx2;
    ctx1.t = 0.0;
    ctx1.position << 7e6, 0.0, 0.0;
    ctx1.velocity = Eigen::Vector3d::Zero();
    
    ctx2.t = 0.0;
    ctx2.position << 14e6, 0.0, 0.0;  // Double the radius
    ctx2.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a1 = gravity_.compute_acceleration(ctx1);
    Eigen::Vector3d a2 = gravity_.compute_acceleration(ctx2);
    
    EXPECT_NEAR(a2.norm(), a1.norm() / 4.0, 1e-6);
}

TEST_F(PointMassGravityTest, VelocityIndependent) {
    // Gravity should not depend on velocity
    ForceContext ctx1, ctx2;
    ctx1.t = 0.0;
    ctx1.position << 7e6, 1e6, 2e6;
    ctx1.velocity << 100.0, 200.0, 300.0;
    
    ctx2.t = 0.0;
    ctx2.position = ctx1.position;
    ctx2.velocity << -500.0, 1000.0, -250.0;
    
    Eigen::Vector3d a1 = gravity_.compute_acceleration(ctx1);
    Eigen::Vector3d a2 = gravity_.compute_acceleration(ctx2);
    
    EXPECT_EQ(a1, a2);
}

TEST_F(PointMassGravityTest, SphericalSymmetry) {
    // At same radius, magnitude should be same regardless of direction
    double r = 8e6;
    
    ForceContext ctx1, ctx2, ctx3;
    ctx1.t = ctx2.t = ctx3.t = 0.0;
    ctx1.position << r, 0.0, 0.0;
    ctx2.position << 0.0, r, 0.0;
    ctx3.position << 0.0, 0.0, r;
    
    double a1_mag = gravity_.compute_acceleration(ctx1).norm();
    double a2_mag = gravity_.compute_acceleration(ctx2).norm();
    double a3_mag = gravity_.compute_acceleration(ctx3).norm();
    
    EXPECT_NEAR(a1_mag, a2_mag, 1e-10);
    EXPECT_NEAR(a2_mag, a3_mag, 1e-10);
}

TEST_F(PointMassGravityTest, NearSingularityHandling) {
    // Should handle near-zero radius gracefully
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 0.5, 0.0, 0.0;  // Very close to origin
    ctx.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a = gravity_.compute_acceleration(ctx);
    
    // Should return some finite value, not NaN/Inf
    EXPECT_TRUE(std::isfinite(a.norm()));
}

// ============================================================================
// POINT MASS GRAVITY JACOBIAN TESTS
// ============================================================================

TEST_F(PointMassGravityTest, JacobianPositionNumerical) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 1e6, 2e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    // Numerical Jacobian
    double epsilon = 1e-3;
    Eigen::Matrix3d da_dr_numerical;
    
    for (int i = 0; i < 3; ++i) {
        ForceContext ctx_plus = ctx;
        ForceContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d a_plus = gravity_.compute_acceleration(ctx_plus);
        Eigen::Vector3d a_minus = gravity_.compute_acceleration(ctx_minus);
        
        da_dr_numerical.col(i) = (a_plus - a_minus) / (2.0 * epsilon);
    }
    
    double error = (da_dr - da_dr_numerical).norm();
    EXPECT_LT(error, 1e-6) << "Position Jacobian error: " << error;
}

TEST_F(PointMassGravityTest, JacobianVelocityZero) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 1e6, 2e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    // Gravity doesn't depend on velocity
    EXPECT_DOUBLE_EQ(da_dv.norm(), 0.0);
}

TEST_F(PointMassGravityTest, JacobianSymmetric) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 1e6, 2e6;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    // Position Jacobian should be symmetric
    Eigen::Matrix3d da_dr_transpose = da_dr.transpose();
    EXPECT_LT((da_dr - da_dr_transpose).norm(), 1e-12);
}

TEST_F(PointMassGravityTest, JacobianStructure) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 0.0, 0.0;  // On x-axis
    ctx.velocity = Eigen::Vector3d::Zero();
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    // For position on x-axis: r = [r, 0, 0]
    // ∂a/∂r = -μ/r³·I + 3μ/r⁵·(r⊗r)
    double r = ctx.position(0);
    double mu_r3 = EARTH_MU / (r * r * r);
    double mu_r5 = EARTH_MU / (r * r * r * r * r);
    
    // Expected structure
    EXPECT_NEAR(da_dr(0, 0), -mu_r3 + 3*mu_r5*r*r, 1e-8);
    EXPECT_NEAR(da_dr(1, 1), -mu_r3, 1e-8);
    EXPECT_NEAR(da_dr(2, 2), -mu_r3, 1e-8);
    EXPECT_NEAR(da_dr(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(da_dr(0, 2), 0.0, 1e-10);
    EXPECT_NEAR(da_dr(1, 2), 0.0, 1e-10);
}

TEST_F(PointMassGravityTest, JacobianEigenvalues) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 1e6, 2e6;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    // Eigenvalues of ∂a/∂r
    // One eigenvalue in radial direction: 2μ/r³
    // Two eigenvalues in tangential directions: -μ/r³
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(da_dr);
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    
    double r = ctx.position.norm();
    double mu_r3 = EARTH_MU / (r * r * r);
    
    // Sort eigenvalues
    std::vector<double> eigs = {eigenvalues(0), eigenvalues(1), eigenvalues(2)};
    std::sort(eigs.begin(), eigs.end());
    
    EXPECT_NEAR(eigs[0], -mu_r3, 1e-6);
    EXPECT_NEAR(eigs[1], -mu_r3, 1e-6);
    EXPECT_NEAR(eigs[2], 2*mu_r3, 1e-6);
}

// ============================================================================
// J2 GRAVITY TESTS
// ============================================================================

class J2GravityTest : public ::testing::Test {
protected:
    J2Gravity gravity_{EARTH_MU, EARTH_J2, EARTH_RADIUS};
};

TEST_F(J2GravityTest, ReducesToPointMassAtEquator) {
    // At equator (z=0), J2 perturbation should be small
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 0.0, 0.0;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a_j2 = gravity_.compute_acceleration(ctx);
    
    // Point mass only
    PointMassGravity pm_gravity(EARTH_MU);
    Eigen::Vector3d a_pm = pm_gravity.compute_acceleration(ctx);
    
    // J2 effect should be small at equator
    double relative_diff = (a_j2 - a_pm).norm() / a_pm.norm();
    EXPECT_LT(relative_diff, 0.01);  // Within 1%
}

TEST_F(J2GravityTest, OblatePerturbation) {
    // J2 causes oblate perturbation (flattening at poles)
    // At poles, radial acceleration should be stronger
    // At equator, tangential effects dominate
    
    ForceContext ctx_pole, ctx_equator;
    ctx_pole.t = ctx_equator.t = 0.0;
    double r = 7e6;
    ctx_pole.position << 0.0, 0.0, r;
    ctx_equator.position << r, 0.0, 0.0;
    
    Eigen::Vector3d a_pole = gravity_.compute_acceleration(ctx_pole);
    Eigen::Vector3d a_equator = gravity_.compute_acceleration(ctx_equator);
    
    // At pole, perturbation increases radial acceleration
    PointMassGravity pm_gravity(EARTH_MU);
    Eigen::Vector3d a_pm_pole = pm_gravity.compute_acceleration(ctx_pole);
    
    EXPECT_GT(a_pole.norm(), a_pm_pole.norm());
}

TEST_F(J2GravityTest, ZonalHarmonicSymmetry) {
    // J2 is axisymmetric about z-axis
    // Same latitude should give same perturbation magnitude
    double r = 7e6;
    double z = 3e6;
    double rho = std::sqrt(r*r - z*z);
    
    ForceContext ctx1, ctx2;
    ctx1.t = ctx2.t = 0.0;
    ctx1.position << rho, 0.0, z;
    ctx2.position << 0.0, rho, z;
    
    Eigen::Vector3d a1 = gravity_.compute_acceleration(ctx1);
    Eigen::Vector3d a2 = gravity_.compute_acceleration(ctx2);
    
    // Magnitudes should be equal
    EXPECT_NEAR(a1.norm(), a2.norm(), 1e-8);
}

TEST_F(J2GravityTest, PerturbationMagnitude) {
    // J2 perturbation should be small compared to point mass
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 5e6, 3e6, 4e6;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a_j2 = gravity_.compute_acceleration(ctx);
    
    PointMassGravity pm_gravity(EARTH_MU);
    Eigen::Vector3d a_pm = pm_gravity.compute_acceleration(ctx);
    
    double perturbation = (a_j2 - a_pm).norm() / a_pm.norm();
    
    // J2 perturbation typically < 0.2%
    // At this position (r ≈ 7.07 Mm, significant z-component), 
    // perturbation is ~0.123%
    EXPECT_LT(perturbation, 0.002);  // Less than 0.2%
}

TEST_F(J2GravityTest, IncreasingWithAltitude) {
    // J2 effect decreases with altitude (Re/r)²
    ForceContext ctx_low, ctx_high;
    ctx_low.t = ctx_high.t = 0.0;
    ctx_low.position << 7e6, 0.0, 0.0;
    ctx_high.position << 14e6, 0.0, 0.0;
    
    PointMassGravity pm_gravity(EARTH_MU);
    
    Eigen::Vector3d a_j2_low = gravity_.compute_acceleration(ctx_low);
    Eigen::Vector3d a_pm_low = pm_gravity.compute_acceleration(ctx_low);
    double pert_low = (a_j2_low - a_pm_low).norm() / a_pm_low.norm();
    
    Eigen::Vector3d a_j2_high = gravity_.compute_acceleration(ctx_high);
    Eigen::Vector3d a_pm_high = pm_gravity.compute_acceleration(ctx_high);
    double pert_high = (a_j2_high - a_pm_high).norm() / a_pm_high.norm();
    
    // Perturbation should decrease with altitude
    EXPECT_LT(pert_high, pert_low);
}

TEST_F(J2GravityTest, VelocityIndependent) {
    ForceContext ctx1, ctx2;
    ctx1.t = ctx2.t = 0.0;
    ctx1.position << 7e6, 1e6, 2e6;
    ctx1.velocity << 100.0, 200.0, 300.0;
    ctx2.position = ctx1.position;
    ctx2.velocity << -500.0, 1000.0, -250.0;
    
    Eigen::Vector3d a1 = gravity_.compute_acceleration(ctx1);
    Eigen::Vector3d a2 = gravity_.compute_acceleration(ctx2);
    
    EXPECT_EQ(a1, a2);
}

// ============================================================================
// J2 GRAVITY JACOBIAN TESTS
// ============================================================================

TEST_F(J2GravityTest, JacobianPositionNumerical) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 1e6, 2e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    // Numerical Jacobian
    double epsilon = 1e-2;  // Larger epsilon for J2 (higher order terms)
    Eigen::Matrix3d da_dr_numerical;
    
    for (int i = 0; i < 3; ++i) {
        ForceContext ctx_plus = ctx;
        ForceContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d a_plus = gravity_.compute_acceleration(ctx_plus);
        Eigen::Vector3d a_minus = gravity_.compute_acceleration(ctx_minus);
        
        da_dr_numerical.col(i) = (a_plus - a_minus) / (2.0 * epsilon);
    }
    
    double error = (da_dr - da_dr_numerical).norm();
    EXPECT_LT(error, 1e-5) << "Position Jacobian error: " << error;
}

TEST_F(J2GravityTest, JacobianVelocityZero) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 1e6, 2e6;
    ctx.velocity << 100.0, 200.0, 300.0;
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    EXPECT_DOUBLE_EQ(da_dv.norm(), 0.0);
}

TEST_F(J2GravityTest, JacobianSymmetric) {
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 1e6, 2e6;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    auto [da_dr, da_dv] = gravity_.compute_jacobian(ctx);
    
    Eigen::Matrix3d da_dr_transpose = da_dr.transpose();
    double symmetry_error = (da_dr - da_dr_transpose).norm();
    EXPECT_LT(symmetry_error, 1e-10) << "Symmetry error: " << symmetry_error;
}

TEST_F(J2GravityTest, JacobianConvergesToPointMass) {
    // At high altitudes, J2 Jacobian should approach point mass Jacobian
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 40e6, 5e6, 10e6;  // High altitude
    ctx.velocity = Eigen::Vector3d::Zero();
    
    auto [da_dr_j2, da_dv_j2] = gravity_.compute_jacobian(ctx);
    
    PointMassGravity pm_gravity(EARTH_MU);
    auto [da_dr_pm, da_dv_pm] = pm_gravity.compute_jacobian(ctx);
    
    double relative_diff = (da_dr_j2 - da_dr_pm).norm() / da_dr_pm.norm();
    EXPECT_LT(relative_diff, 0.01);  // Within 1% at high altitude
}

TEST_F(J2GravityTest, JacobianPoleEquatorDifference) {
    // Jacobian should differ between pole and equator
    ForceContext ctx_pole, ctx_equator;
    ctx_pole.t = ctx_equator.t = 0.0;
    double r = 7e6;
    ctx_pole.position << 0.0, 0.0, r;
    ctx_equator.position << r, 0.0, 0.0;
    
    auto [da_dr_pole, _1] = gravity_.compute_jacobian(ctx_pole);
    auto [da_dr_equator, _2] = gravity_.compute_jacobian(ctx_equator);
    
    // Jacobians should be different
    EXPECT_GT((da_dr_pole - da_dr_equator).norm(), 1e-10);
}

// ============================================================================
// COMPARISON TESTS
// ============================================================================

TEST(GravityComparisonTest, J2SmallPerturbation) {
    // J2 should be a small perturbation to point mass
    PointMassGravity pm_gravity(EARTH_MU);
    J2Gravity j2_gravity(EARTH_MU, EARTH_J2, EARTH_RADIUS);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 7e6, 2e6, 3e6;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a_pm = pm_gravity.compute_acceleration(ctx);
    Eigen::Vector3d a_j2 = j2_gravity.compute_acceleration(ctx);
    
    double perturbation = (a_j2 - a_pm).norm() / a_pm.norm();
    EXPECT_LT(perturbation, 0.002);  // Less than 0.2%
}

TEST(GravityComparisonTest, JacobianConsistency) {
    // Both models should have similar Jacobian structure
    PointMassGravity pm_gravity(EARTH_MU);
    J2Gravity j2_gravity(EARTH_MU, EARTH_J2, EARTH_RADIUS);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << 20e6, 5e6, 8e6;  // High altitude
    ctx.velocity = Eigen::Vector3d::Zero();
    
    auto [da_dr_pm, _1] = pm_gravity.compute_jacobian(ctx);
    auto [da_dr_j2, _2] = j2_gravity.compute_jacobian(ctx);
    
    // At high altitude, should be very similar
    double relative_diff = (da_dr_j2 - da_dr_pm).norm() / da_dr_pm.norm();
    EXPECT_LT(relative_diff, 0.05);  // Within 5%
}

// ============================================================================
// ORBITAL MECHANICS VALIDATION
// ============================================================================

TEST(GravityOrbitalTest, CircularOrbitAcceleration) {
    PointMassGravity gravity(EARTH_MU);
    
    // Circular orbit at 400 km altitude
    double r = EARTH_RADIUS + 400e3;
    double v = std::sqrt(EARTH_MU / r);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << r, 0.0, 0.0;
    ctx.velocity << 0.0, v, 0.0;
    
    Eigen::Vector3d a = gravity.compute_acceleration(ctx);
    
    // Centripetal acceleration
    double a_centripetal = v * v / r;
    EXPECT_NEAR(a.norm(), a_centripetal, 1e-3);
    
    // Should point toward Earth
    EXPECT_LT(a.dot(ctx.position), 0.0);
}

TEST(GravityOrbitalTest, EscapeVelocity) {
    PointMassGravity gravity(EARTH_MU);
    
    double r = EARTH_RADIUS + 200e3;
    double v_escape = std::sqrt(2.0 * EARTH_MU / r);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << r, 0.0, 0.0;
    ctx.velocity << 0.0, v_escape, 0.0;
    
    Eigen::Vector3d a = gravity.compute_acceleration(ctx);
    
    // At escape velocity, specific energy = 0
    // E = v²/2 - μ/r = 0
    double kinetic = 0.5 * v_escape * v_escape;
    double potential = EARTH_MU / r;
    EXPECT_NEAR(kinetic, potential, 1e-3);
}

TEST(GravityOrbitalTest, Geostationary) {
    PointMassGravity gravity(EARTH_MU);
    
    // Geostationary orbit radius
    constexpr double OMEGA_EARTH = 7.2921159e-5;  // rad/s
    double r_geo = std::pow(EARTH_MU / (OMEGA_EARTH * OMEGA_EARTH), 1.0/3.0);
    
    ForceContext ctx;
    ctx.t = 0.0;
    ctx.position << r_geo, 0.0, 0.0;
    ctx.velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d a = gravity.compute_acceleration(ctx);
    
    // Centripetal acceleration for geostationary orbit
    double a_expected = OMEGA_EARTH * OMEGA_EARTH * r_geo;
    EXPECT_NEAR(a.norm(), a_expected, 1e-3);
}
