#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <transforms/coord_transforms.hpp>

TEST(CoordTransformsTest, ECEFtoECIAndBack) {
    // Position (m) and velocity (m/s) in ECEF: point on equator with Earth's rotation velocity
    auto state_ecef = (Eigen::VectorXd(6) << 
        6378137.0, 0.0, 0.0,
        0.0, 465.101357, 0.0
    ).finished();

    double time_seconds = 3600.0; // 1 hour after epoch

    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);
    auto state_eci = coord_transforms.ecef_to_eci(state_ecef, time_seconds);
    auto state_ecef_converted = coord_transforms.eci_to_ecef(state_eci, time_seconds);

    // Compare all 6 components (pos [0..2], vel [3..5])
    double tol_pos = 1e-6;   // meters
    double tol_vel = 1e-6;   // meters/second

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(state_ecef(i), state_ecef_converted(i), tol_pos);
    }
    for (int i = 3; i < 6; ++i) {
        EXPECT_NEAR(state_ecef(i), state_ecef_converted(i), tol_vel);
    }
}

TEST(CoordTransformsTest, ECItoECEFAndBack) {
    // Position (m) and velocity (m/s) in ECI: circular orbit example
    double r = 7000e3;
    // circular orbital speed ~ sqrt(mu / r)
    const double mu = 3.986004418e14;
    double v_circ = std::sqrt(mu / r);

    auto state_eci = (Eigen::VectorXd(6) << 
        r, 0.0, 0.0,
        0.0, v_circ, 0.0
    ).finished();

    double time_seconds = 7200.0; // 2 hours after epoch

    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);
    auto state_ecef = coord_transforms.eci_to_ecef(state_eci, time_seconds);
    auto state_eci_converted = coord_transforms.ecef_to_eci(state_ecef, time_seconds);

    double tol_pos = 1e-6;   // meters
    double tol_vel = 1e-6;   // meters/second

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_pos);
    }
    for (int i = 3; i < 6; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_vel);
    }
}
