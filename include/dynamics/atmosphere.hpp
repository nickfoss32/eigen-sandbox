#pragma once

#include <cmath>

namespace dynamics {
namespace atmosphere {

/// @brief Simple exponential atmosphere model
/// @param altitude Altitude above sea level (m)
/// @return Atmospheric density (kg/m³)
inline double get_density(double altitude) {
    // Sea level density
    constexpr double rho_0 = 1.225;  // kg/m³
    
    // Scale height (exponential decrease)
    constexpr double H = 8500.0;  // m
    
    if (altitude < 0.0) {
        return rho_0;  // Below sea level
    }
    
    // Exponential atmosphere: ρ(h) = ρ₀ * e^(-h/H)
    return rho_0 * std::exp(-altitude / H);
}

/// @brief US Standard Atmosphere 1976 (more accurate)
/// @param altitude Altitude above sea level (m)
/// @return Atmospheric density (kg/m³)
inline double get_density_us76(double altitude) {
    // Piecewise model for different atmospheric layers
    
    if (altitude < 0.0) {
        return 1.225;
    }
    
    // Troposphere (0-11 km)
    if (altitude < 11000.0) {
        double T = 288.15 - 0.0065 * altitude;  // Temperature (K)
        double p = 101325.0 * std::pow(T / 288.15, 5.2561);  // Pressure (Pa)
        return p / (287.05 * T);  // Density from ideal gas law
    }
    
    // Lower Stratosphere (11-25 km)
    if (altitude < 25000.0) {
        double T = 216.65;  // Isothermal layer
        double p = 22632.0 * std::exp(-0.0001577 * (altitude - 11000.0));
        return p / (287.05 * T);
    }
    
    // Upper Stratosphere (25-47 km)
    if (altitude < 47000.0) {
        double T = 216.65 + 0.003 * (altitude - 25000.0);
        double p = 2488.4 * std::pow(T / 216.65, -11.388);
        return p / (287.05 * T);
    }
    
    // Above 47 km - use exponential decay
    return 0.00143 * std::exp(-(altitude - 47000.0) / 7200.0);
}

} // namespace atmosphere
} // namespace dynamics
