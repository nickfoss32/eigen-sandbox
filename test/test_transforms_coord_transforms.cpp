#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <transforms/coord_transforms.hpp>

// Test 3-state (position only) ECEF to ECI and back
TEST(CoordTransformsTest, ECEFtoECIAndBack_3State) {
    // Position (m) in ECEF: point on equator
    auto state_ecef = (Eigen::VectorXd(3) << 
        6378137.0, 0.0, 0.0
    ).finished();

    double time_seconds = 3600.0; // 1 hour after epoch

    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);
    auto state_eci = coord_transforms.ecef_to_eci(state_ecef, time_seconds);
    auto state_ecef_converted = coord_transforms.eci_to_ecef(state_eci, time_seconds);

    ASSERT_EQ(state_eci.size(), 3);
    ASSERT_EQ(state_ecef_converted.size(), 3);

    double tol_pos = 1e-6;   // meters

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(state_ecef(i), state_ecef_converted(i), tol_pos);
    }
}

// Test 6-state (position + velocity) ECEF to ECI and back
TEST(CoordTransformsTest, ECEFtoECIAndBack_6State) {
    // Position (m) and velocity (m/s) in ECEF: point on equator with Earth's rotation velocity
    auto state_ecef = (Eigen::VectorXd(6) << 
        6378137.0, 0.0, 0.0,
        0.0, 465.101357, 0.0
    ).finished();

    double time_seconds = 3600.0; // 1 hour after epoch

    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);
    auto state_eci = coord_transforms.ecef_to_eci(state_ecef, time_seconds);
    auto state_ecef_converted = coord_transforms.eci_to_ecef(state_eci, time_seconds);

    ASSERT_EQ(state_eci.size(), 6);
    ASSERT_EQ(state_ecef_converted.size(), 6);

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

// Test 9-state (position + velocity + acceleration) ECEF to ECI and back
TEST(CoordTransformsTest, ECEFtoECIAndBack_9State) {
    // Position (m), velocity (m/s), and acceleration (m/s²) in ECEF
    auto state_ecef = (Eigen::VectorXd(9) << 
        6378137.0, 0.0, 0.0,           // position
        0.0, 465.101357, 0.0,          // velocity
        -0.034, 0.0, 0.0               // acceleration (centripetal)
    ).finished();

    double time_seconds = 3600.0; // 1 hour after epoch

    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);
    auto state_eci = coord_transforms.ecef_to_eci(state_ecef, time_seconds);
    auto state_ecef_converted = coord_transforms.eci_to_ecef(state_eci, time_seconds);

    ASSERT_EQ(state_eci.size(), 9);
    ASSERT_EQ(state_ecef_converted.size(), 9);

    double tol_pos = 1e-6;   // meters
    double tol_vel = 1e-6;   // meters/second
    double tol_acc = 1e-9;   // meters/second²

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(state_ecef(i), state_ecef_converted(i), tol_pos);
    }
    for (int i = 3; i < 6; ++i) {
        EXPECT_NEAR(state_ecef(i), state_ecef_converted(i), tol_vel);
    }
    for (int i = 6; i < 9; ++i) {
        EXPECT_NEAR(state_ecef(i), state_ecef_converted(i), tol_acc);
    }
}

// Test 3-state (position only) ECI to ECEF and back
TEST(CoordTransformsTest, ECItoECEFAndBack_3State) {
    // Position (m) in ECI
    double r = 7000e3;
    auto state_eci = (Eigen::VectorXd(3) << 
        r, 0.0, 0.0
    ).finished();

    double time_seconds = 7200.0; // 2 hours after epoch

    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);
    auto state_ecef = coord_transforms.eci_to_ecef(state_eci, time_seconds);
    auto state_eci_converted = coord_transforms.ecef_to_eci(state_ecef, time_seconds);

    ASSERT_EQ(state_ecef.size(), 3);
    ASSERT_EQ(state_eci_converted.size(), 3);

    double tol_pos = 1e-6;   // meters

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_pos);
    }
}

// Test 6-state (position + velocity) ECI to ECEF and back
TEST(CoordTransformsTest, ECItoECEFAndBack_6State) {
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

    ASSERT_EQ(state_ecef.size(), 6);
    ASSERT_EQ(state_eci_converted.size(), 6);

    double tol_pos = 1e-6;   // meters
    double tol_vel = 1e-6;   // meters/second

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_pos);
    }
    for (int i = 3; i < 6; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_vel);
    }
}

// Test 9-state (position + velocity + acceleration) ECI to ECEF and back
TEST(CoordTransformsTest, ECItoECEFAndBack_9State) {
    // Position (m), velocity (m/s), and acceleration (m/s²) in ECI: circular orbit
    double r = 7000e3;
    const double mu = 3.986004418e14;
    double v_circ = std::sqrt(mu / r);
    double a_centripetal = v_circ * v_circ / r;  // Centripetal acceleration

    auto state_eci = (Eigen::VectorXd(9) << 
        r, 0.0, 0.0,                      // position
        0.0, v_circ, 0.0,                 // velocity
        -a_centripetal, 0.0, 0.0          // acceleration (toward Earth)
    ).finished();

    double time_seconds = 7200.0; // 2 hours after epoch

    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);
    auto state_ecef = coord_transforms.eci_to_ecef(state_eci, time_seconds);
    auto state_eci_converted = coord_transforms.ecef_to_eci(state_ecef, time_seconds);

    ASSERT_EQ(state_ecef.size(), 9);
    ASSERT_EQ(state_eci_converted.size(), 9);

    double tol_pos = 1e-6;   // meters
    double tol_vel = 1e-6;   // meters/second
    double tol_acc = 1e-9;   // meters/second²

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_pos);
    }
    for (int i = 3; i < 6; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_vel);
    }
    for (int i = 6; i < 9; ++i) {
        EXPECT_NEAR(state_eci(i), state_eci_converted(i), tol_acc);
    }
}

// Test LLA to ECEF conversion
TEST(CoordTransformsTest, LLAtoECEF) {
    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);

    // Test case 1: Point on equator at prime meridian
    double lat1 = 0.0;    // degrees
    double lon1 = 0.0;    // degrees
    double alt1 = 0.0;    // meters
    auto ecef1 = coord_transforms.lla_to_ecef(lat1, lon1, alt1);
    
    // WGS84 semi-major axis
    double a = 6378137.0;
    EXPECT_NEAR(ecef1(0), a, 1.0);      // x ≈ 6378137 m
    EXPECT_NEAR(ecef1(1), 0.0, 1.0);    // y ≈ 0
    EXPECT_NEAR(ecef1(2), 0.0, 1.0);    // z ≈ 0

    // Test case 2: Point on equator at 90° East
    double lat2 = 0.0;
    double lon2 = 90.0;
    double alt2 = 0.0;
    auto ecef2 = coord_transforms.lla_to_ecef(lat2, lon2, alt2);
    
    EXPECT_NEAR(ecef2(0), 0.0, 1.0);    // x ≈ 0
    EXPECT_NEAR(ecef2(1), a, 1.0);      // y ≈ 6378137 m
    EXPECT_NEAR(ecef2(2), 0.0, 1.0);    // z ≈ 0

    // Test case 3: North pole
    double lat3 = 90.0;
    double lon3 = 0.0;
    double alt3 = 0.0;
    auto ecef3 = coord_transforms.lla_to_ecef(lat3, lon3, alt3);
    
    // WGS84 semi-minor axis
    double b = 6356752.314245;
    EXPECT_NEAR(ecef3(0), 0.0, 1.0);    // x ≈ 0
    EXPECT_NEAR(ecef3(1), 0.0, 1.0);    // y ≈ 0
    EXPECT_NEAR(ecef3(2), b, 1.0);      // z ≈ 6356752 m

    // Test case 4: Arbitrary location with altitude (New York City approximate)
    double lat4 = 40.7128;   // degrees North
    double lon4 = -74.0060;  // degrees West
    double alt4 = 10.0;      // meters
    auto ecef4 = coord_transforms.lla_to_ecef(lat4, lon4, alt4);
    
    // Just verify it produces reasonable values (not NaN, not zero)
    EXPECT_FALSE(std::isnan(ecef4(0)));
    EXPECT_FALSE(std::isnan(ecef4(1)));
    EXPECT_FALSE(std::isnan(ecef4(2)));
    EXPECT_GT(ecef4.norm(), 6.3e6);  // Should be > Earth's radius
    EXPECT_LT(ecef4.norm(), 6.4e6);  // Should be < max Earth's radius
}

// Test ECEF to LLA conversion
TEST(CoordTransformsTest, ECEFtoLLA) {
    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);

    // Test case 1: Point on equator at prime meridian
    Eigen::Vector3d ecef1(6378137.0, 0.0, 0.0);
    auto lla1 = coord_transforms.ecef_to_lla(ecef1);
    
    EXPECT_NEAR(lla1(0), 0.0, 1e-6);    // latitude ≈ 0°
    EXPECT_NEAR(lla1(1), 0.0, 1e-6);    // longitude ≈ 0°
    EXPECT_NEAR(lla1(2), 0.0, 1.0);     // altitude ≈ 0 m

    // Test case 2: Point on equator at 90° East
    Eigen::Vector3d ecef2(0.0, 6378137.0, 0.0);
    auto lla2 = coord_transforms.ecef_to_lla(ecef2);
    
    EXPECT_NEAR(lla2(0), 0.0, 1e-6);    // latitude ≈ 0°
    EXPECT_NEAR(lla2(1), 90.0, 1e-6);   // longitude ≈ 90°
    EXPECT_NEAR(lla2(2), 0.0, 1.0);     // altitude ≈ 0 m

    // Test case 3: North pole
    Eigen::Vector3d ecef3(0.0, 0.0, 6356752.314245);
    auto lla3 = coord_transforms.ecef_to_lla(ecef3);
    
    EXPECT_NEAR(lla3(0), 90.0, 1e-6);   // latitude ≈ 90°
    EXPECT_NEAR(lla3(2), 0.0, 1.0);     // altitude ≈ 0 m
    // Longitude is undefined at poles

    // Test case 4: Point with altitude
    Eigen::Vector3d ecef4(6378137.0 + 1000.0, 0.0, 0.0);
    auto lla4 = coord_transforms.ecef_to_lla(ecef4);
    
    EXPECT_NEAR(lla4(0), 0.0, 1e-6);      // latitude ≈ 0°
    EXPECT_NEAR(lla4(1), 0.0, 1e-6);      // longitude ≈ 0°
    EXPECT_NEAR(lla4(2), 1000.0, 1.0);    // altitude ≈ 1000 m
}

// Test round-trip LLA to ECEF to LLA
TEST(CoordTransformsTest, LLAtoECEFtoLLA) {
    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);

    // Test multiple locations
    std::vector<std::tuple<double, double, double>> test_locations = {
        {0.0, 0.0, 0.0},           // Equator, prime meridian, sea level
        {40.7128, -74.0060, 10.0}, // New York City
        {51.5074, -0.1278, 11.0},  // London
        {-33.8688, 151.2093, 58.0},// Sydney
        {35.6762, 139.6503, 40.0}, // Tokyo
        {-90.0, 0.0, 0.0},         // South pole
        {89.9, 0.0, 0.0}           // Near north pole (avoid singularity)
    };

    for (const auto& [lat, lon, alt] : test_locations) {
        auto ecef = coord_transforms.lla_to_ecef(lat, lon, alt);
        auto lla_result = coord_transforms.ecef_to_lla(ecef);

        EXPECT_NEAR(lla_result(0), lat, 1e-6) << "Latitude mismatch for (" << lat << ", " << lon << ", " << alt << ")";
        EXPECT_NEAR(lla_result(1), lon, 1e-6) << "Longitude mismatch for (" << lat << ", " << lon << ", " << alt << ")";
        EXPECT_NEAR(lla_result(2), alt, 1e-3) << "Altitude mismatch for (" << lat << ", " << lon << ", " << alt << ")";
    }
}

// Test round-trip ECEF to LLA to ECEF
TEST(CoordTransformsTest, ECEFtoLLAtoECEF) {
    auto coord_transforms = transforms::CoordTransforms(IERS_EOP_FILE);

    // Test multiple ECEF positions
    std::vector<Eigen::Vector3d> test_positions = {
        Eigen::Vector3d(6378137.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 6378137.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 6356752.314245),
        Eigen::Vector3d(4510731.0, -4510731.0, 3816878.0),  // Arbitrary point
        Eigen::Vector3d(3194434.0, -3194434.0, 4487349.0)   // Another arbitrary point
    };

    for (const auto& ecef : test_positions) {
        auto lla = coord_transforms.ecef_to_lla(ecef);
        auto ecef_result = coord_transforms.lla_to_ecef(lla(0), lla(1), lla(2));

        EXPECT_NEAR(ecef_result(0), ecef(0), 1e-3) << "X mismatch for ECEF (" << ecef.transpose() << ")";
        EXPECT_NEAR(ecef_result(1), ecef(1), 1e-3) << "Y mismatch for ECEF (" << ecef.transpose() << ")";
        EXPECT_NEAR(ecef_result(2), ecef(2), 1e-3) << "Z mismatch for ECEF (" << ecef.transpose() << ")";
    }
}
