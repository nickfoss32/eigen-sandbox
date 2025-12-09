#include <gtest/gtest.h>
#include "dynamics/atmosphere.hpp"
#include <cmath>

using namespace dynamics::atmosphere;

// ============================================================================
// TEST: Simple Exponential Atmosphere Model
// ============================================================================

TEST(AtmosphereTest, ExponentialModelSeaLevel) {
    // At sea level, density should be ~1.225 kg/m³
    double rho = get_density(0.0);
    EXPECT_NEAR(rho, 1.225, 1e-6);
}

TEST(AtmosphereTest, ExponentialModelBelowSeaLevel) {
    // Below sea level, should clamp to sea level density
    double rho = get_density(-100.0);
    EXPECT_NEAR(rho, 1.225, 1e-6);
}

TEST(AtmosphereTest, ExponentialModelScaleHeight) {
    // At one scale height (8.5 km), density should be ~1/e of sea level
    double rho = get_density(8500.0);
    double expected = 1.225 / std::exp(1.0);
    EXPECT_NEAR(rho, expected, 1e-6);
}

TEST(AtmosphereTest, ExponentialModelLowEarthOrbit) {
    // At 400 km (ISS altitude), density should be very small
    double rho = get_density(400e3);
    EXPECT_LT(rho, 1e-10);
    EXPECT_GT(rho, 0.0);
}

TEST(AtmosphereTest, ExponentialModelMonotonic) {
    // Density should always decrease with altitude
    double rho_100km = get_density(100e3);
    double rho_200km = get_density(200e3);
    double rho_300km = get_density(300e3);
    
    EXPECT_GT(rho_100km, rho_200km);
    EXPECT_GT(rho_200km, rho_300km);
}

// ============================================================================
// TEST: US Standard Atmosphere 1976
// ============================================================================

TEST(AtmosphereUS76Test, SeaLevel) {
    // At sea level: T = 288.15 K, p = 101325 Pa
    // ρ = p/(R·T) = 101325/(287.05·288.15) ≈ 1.225 kg/m³
    double rho = get_density_us76(0.0);
    EXPECT_NEAR(rho, 1.225, 0.001);  // Within 0.1%
}

TEST(AtmosphereUS76Test, BelowSeaLevel) {
    // Below sea level, should return sea level density
    double rho = get_density_us76(-500.0);
    EXPECT_NEAR(rho, 1.225, 0.001);
}

TEST(AtmosphereUS76Test, Troposphere) {
    // At 5 km altitude (middle of troposphere)
    // Expected: ~0.736 kg/m³
    double rho = get_density_us76(5000.0);
    EXPECT_NEAR(rho, 0.736, 0.05);  // Within 5%
}

TEST(AtmosphereUS76Test, TropopauseBoundary) {
    // At 11 km (tropopause boundary)
    // Temperature becomes isothermal at 216.65 K
    double rho = get_density_us76(11000.0);
    EXPECT_NEAR(rho, 0.3639, 0.01);  // Expected value from standard tables
}

TEST(AtmosphereUS76Test, LowerStratosphere) {
    // At 20 km altitude (lower stratosphere, isothermal)
    // Expected: ~0.0880 kg/m³
    double rho = get_density_us76(20000.0);
    EXPECT_NEAR(rho, 0.0880, 0.01);
}

TEST(AtmosphereUS76Test, UpperStratosphere) {
    // At 30 km altitude (upper stratosphere, temperature increasing)
    // Expected: ~0.0184 kg/m³
    double rho = get_density_us76(30000.0);
    EXPECT_NEAR(rho, 0.0184, 0.005);
}

TEST(AtmosphereUS76Test, StratopauseBoundary) {
    // At 47 km (stratopause boundary)
    // From upper stratosphere formula:
    // T = 216.65 + 0.003 × (47000 - 25000) = 282.65 K
    // p = 2488.4 × (282.65/216.65)^(-11.388) ≈ 120.5 Pa
    // ρ = p/(R·T) = 120.5/(287.05×282.65) ≈ 0.001484 kg/m³
    double rho_47km = get_density_us76(47000.0);
    EXPECT_NEAR(rho_47km, 0.001484, 0.0002);
}

TEST(AtmosphereUS76Test, Mesosphere) {
    // At 60 km altitude (mesosphere, exponential decay)
    // Expected: ~3e-4 kg/m³
    double rho = get_density_us76(60000.0);
    EXPECT_NEAR(rho, 3e-4, 1e-4);
}

TEST(AtmosphereUS76Test, LowEarthOrbit) {
    // At 200 km (LEO)
    // Expected: ~6e-9 kg/m³
    double rho = get_density_us76(200e3);
    EXPECT_LT(rho, 1e-8);
    EXPECT_GT(rho, 1e-10);
}

TEST(AtmosphereUS76Test, ISSAltitude) {
    // At 400 km (ISS altitude)
    // With improved multi-scale-height model: ~1e-10 kg/m³
    double rho = get_density_us76(400e3);
    EXPECT_LT(rho, 5e-10);   // Not too high
    EXPECT_GT(rho, 5e-12);   // Not too low
}

TEST(AtmosphereUS76Test, Monotonic) {
    // Density should always decrease with altitude
    std::vector<double> altitudes = {0, 5e3, 11e3, 20e3, 30e3, 47e3, 60e3, 100e3, 200e3};
    
    for (size_t i = 1; i < altitudes.size(); ++i) {
        double rho_lower = get_density_us76(altitudes[i-1]);
        double rho_upper = get_density_us76(altitudes[i]);
        EXPECT_GT(rho_lower, rho_upper) 
            << "Density not monotonic between " << altitudes[i-1]/1e3 
            << " km and " << altitudes[i]/1e3 << " km";
    }
}

TEST(AtmosphereUS76Test, Continuity) {
    // Check continuity at layer boundaries (no jumps)
    std::vector<double> boundaries = {11000.0, 25000.0, 47000.0};
    
    for (double boundary : boundaries) {
        double rho_below = get_density_us76(boundary - 0.1);
        double rho_above = get_density_us76(boundary + 0.1);
        
        // Should be continuous (within 1%)
        double relative_diff = std::abs(rho_above - rho_below) / rho_below;
        EXPECT_LT(relative_diff, 0.01) 
            << "Discontinuity at " << boundary/1e3 << " km boundary";
    }
}

// ============================================================================
// TEST: Comparison Between Models
// ============================================================================

TEST(AtmosphereComparisonTest, SeaLevelAgreement) {
    // Both models should agree at sea level
    double rho_exp = get_density(0.0);
    double rho_us76 = get_density_us76(0.0);
    EXPECT_NEAR(rho_exp, rho_us76, 0.001);
}

TEST(AtmosphereComparisonTest, LowAltitudeComparison) {
    // At low altitudes (< 10 km), models should be reasonably close
    for (double h = 0; h < 10000; h += 1000) {
        double rho_exp = get_density(h);
        double rho_us76 = get_density_us76(h);
        
        // Within 20% (exponential is simplified)
        double relative_diff = std::abs(rho_exp - rho_us76) / rho_us76;
        EXPECT_LT(relative_diff, 0.20) 
            << "Models diverge too much at " << h/1e3 << " km";
    }
}

TEST(AtmosphereComparisonTest, US76MoreAccurateThanExponential) {
    // US76 should be more accurate in stratosphere
    // Known value at 25 km: ~0.0399 kg/m³
    double rho_us76 = get_density_us76(25000.0);
    double rho_exp = get_density(25000.0);
    
    // US76 should be closer to true value
    double error_us76 = std::abs(rho_us76 - 0.0399);
    double error_exp = std::abs(rho_exp - 0.0399);
    
    EXPECT_LT(error_us76, error_exp);
}

// ============================================================================
// TEST: Physical Properties
// ============================================================================

TEST(AtmospherePhysicsTest, PositiveDensity) {
    // Density should always be positive
    std::vector<double> test_altitudes = {0, 10e3, 50e3, 100e3, 500e3, 1000e3};
    
    for (double h : test_altitudes) {
        EXPECT_GT(get_density(h), 0.0) 
            << "Exponential model gives non-positive density at " << h/1e3 << " km";
        EXPECT_GT(get_density_us76(h), 0.0) 
            << "US76 model gives non-positive density at " << h/1e3 << " km";
    }
}

TEST(AtmospherePhysicsTest, ReasonableMagnitudes) {
    // Check densities are in physically reasonable ranges
    
    // Sea level: ~1 kg/m³
    EXPECT_NEAR(get_density_us76(0), 1.225, 0.1);
    
    // Airliner cruise (10 km): ~0.4 kg/m³
    EXPECT_NEAR(get_density_us76(10e3), 0.4, 0.1);
    
    // Low stratosphere (30 km): ~0.02 kg/m³
    EXPECT_NEAR(get_density_us76(30e3), 0.02, 0.01);
    
    // LEO (400 km): ~1e-10 kg/m³ (order of magnitude)
    // Actual density varies with solar activity, but should be in range
    EXPECT_LT(get_density_us76(400e3), 5e-10);
    EXPECT_GT(get_density_us76(400e3), 1e-12);
}

TEST(AtmospherePhysicsTest, ScaleHeightBehavior) {
    // In exponential model, density should drop by factor of e every scale height
    double H = 8500.0;  // Scale height
    
    for (int n = 0; n < 10; ++n) {
        double h1 = n * H;
        double h2 = (n + 1) * H;
        
        double rho1 = get_density(h1);
        double rho2 = get_density(h2);
        
        double ratio = rho1 / rho2;
        EXPECT_NEAR(ratio, std::exp(1.0), 0.01) 
            << "Scale height behavior violated at " << h1/1e3 << " km";
    }
}

// ============================================================================
// TEST: Edge Cases
// ============================================================================

TEST(AtmosphereEdgeCasesTest, VeryHighAltitude) {
    // At very high altitudes, density should be extremely small but not zero
    double rho = get_density_us76(1000e3);  // 1000 km
    EXPECT_GT(rho, 0.0);
    EXPECT_LT(rho, 1e-15);
}

TEST(AtmosphereEdgeCasesTest, NegativeAltitudeLarge) {
    // Even at large negative altitudes, should clamp to sea level
    double rho = get_density(-10000.0);
    EXPECT_NEAR(rho, 1.225, 1e-6);
    
    rho = get_density_us76(-10000.0);
    EXPECT_NEAR(rho, 1.225, 0.001);
}

TEST(AtmosphereEdgeCasesTest, ZeroAltitude) {
    // Exactly at sea level (h = 0)
    double rho_exp = get_density(0.0);
    double rho_us76 = get_density_us76(0.0);
    
    EXPECT_NEAR(rho_exp, 1.225, 1e-6);
    EXPECT_NEAR(rho_us76, 1.225, 0.001);
}

// ============================================================================
// TEST: Derivative Properties (for Jacobian validation)
// ============================================================================

TEST(AtmosphereDensityGradientTest, NumericalDerivative) {
    // Test that density gradient is negative (decreasing with altitude)
    double h = 10000.0;  // 10 km
    double dh = 1.0;     // 1 meter
    
    double rho_plus = get_density_us76(h + dh);
    double rho_minus = get_density_us76(h - dh);
    double drho_dh = (rho_plus - rho_minus) / (2.0 * dh);
    
    // Gradient should be negative
    EXPECT_LT(drho_dh, 0.0) << "Density should decrease with altitude";
}

TEST(AtmosphereDensityGradientTest, GradientMagnitudeReasonable) {
    // At 10 km, density gradient should be ~-5e-5 kg/m³/m
    double h = 10000.0;
    double dh = 1.0;
    
    double rho_plus = get_density_us76(h + dh);
    double rho_minus = get_density_us76(h - dh);
    double drho_dh = (rho_plus - rho_minus) / (2.0 * dh);
    
    // Order of magnitude check
    EXPECT_GT(std::abs(drho_dh), 1e-6);
    EXPECT_LT(std::abs(drho_dh), 1e-3);
}

// ============================================================================
// TEST: Practical Application Values
// ============================================================================

TEST(AtmospherePracticalTest, CommercialAircraftAltitude) {
    // Typical cruise altitude: 35,000 ft = 10,668 m
    double rho = get_density_us76(10668.0);
    
    // Should be about 0.38 kg/m³
    EXPECT_NEAR(rho, 0.38, 0.05);
}

TEST(AtmospherePracticalTest, SpaceStationAltitude) {
    // ISS orbits at ~408 km
    double rho = get_density_us76(408e3);
    
    EXPECT_LT(rho, 5e-10);   // Upper bound
    EXPECT_GT(rho, 5e-12);   // Lower bound
}

TEST(AtmospherePracticalTest, WeatherBalloonAltitude) {
    // High-altitude balloons reach ~30-40 km
    double rho_30km = get_density_us76(30e3);
    double rho_40km = get_density_us76(40e3);
    
    // Should be in range 0.002 to 0.02 kg/m³
    EXPECT_GT(rho_30km, 0.002);
    EXPECT_LT(rho_30km, 0.02);
    EXPECT_GT(rho_40km, 0.001);
    EXPECT_LT(rho_40km, 0.01);
}

TEST(AtmospherePracticalTest, KarmanLine) {
    // Kármán line (boundary of space): 100 km
    double rho = get_density_us76(100e3);
    
    // Density at 100 km varies by atmosphere model:
    // - Full US76: ~5.6e-7 kg/m³
    // - Simplified exponential: ~9e-7 kg/m³
    // Our implementation uses simplified model above 47 km
    
    // Check order of magnitude is correct (thermosphere)
    EXPECT_GT(rho, 1e-7);   // Not too low
    EXPECT_LT(rho, 2e-6);   // Not too high
    
    // More specific check if using simplified model
    EXPECT_NEAR(rho, 9.09e-7, 3e-7);
}
