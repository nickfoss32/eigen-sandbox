#pragma once

#include <Eigen/Dense>
#include <cmath>

/// @brief Converts latitude, longitude, and altitude to ECEF coordinates using the WGS84 ellipsoid.
/// @param lat Latitude in degrees (positive north, negative south).
/// @param lon Longitude in degrees (positive east, negative west).
/// @param alt Altitude above the WGS84 ellipsoid in meters (default: 0.0).
/// @return ECEF coordinates (x, y, z) in meters as an Eigen::Vector3d.
Eigen::Vector3d lla_to_ecef(double lat, double lon, double alt = 0.0) {
    // WGS84 ellipsoid parameters
    const double a = 6378137.0; // Semi-major axis (meters)
    const double f = 1.0 / 298.257223563; // Flattening
    const double e2 = f * (2.0 - f); // Square of first eccentricity

    // Convert degrees to radians
    const double lat_rad = lat * M_PI / 180.0;
    const double lon_rad = lon * M_PI / 180.0;

    // Radius of curvature in the prime vertical
    const double sin_lat = std::sin(lat_rad);
    const double N = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);

    // ECEF coordinates
    Eigen::Vector3d ecef;
    ecef(0) = (N + alt) * std::cos(lat_rad) * std::cos(lon_rad); // x
    ecef(1) = (N + alt) * std::cos(lat_rad) * std::sin(lon_rad); // y
    ecef(2) = (N * (1.0 - e2) + alt) * std::sin(lat_rad);        // z

    return ecef;
}
