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

/// @brief Converts ECEF coordinates to latitude, longitude, and altitude using the WGS84 ellipsoid.
/// @param ecef ECEF coordinates (x, y, z) in meters as an Eigen::Vector3d.
/// @return LLA coordinates as an Eigen::Vector3d: (latitude in degrees, longitude in degrees, altitude in meters).
Eigen::Vector3d ecef_to_lla(const Eigen::Vector3d& ecef) {
    // WGS84 ellipsoid parameters
    const double a = 6378137.0; // Semi-major axis (meters)
    const double f = 1.0 / 298.257223563; // Flattening
    const double b = a * (1.0 - f); // Semi-minor axis (meters)
    const double e2 = f * (2.0 - f); // Square of first eccentricity
    const double ep2 = e2 / (1.0 - e2); // Square of second eccentricity

    // Extract ECEF coordinates
    const double x = ecef(0);
    const double y = ecef(1);
    const double z = ecef(2);

    // Longitude calculation
    double lon = std::atan2(y, x);

    // Compute the distance from the z-axis
    const double p = std::sqrt(x * x + y * y);

    // Initial guess for latitude
    double lat = std::atan2(z, p * (1.0 - e2));

    // Iterative computation for latitude and altitude
    double sin_lat, N, h;
    const int max_iterations = 10;
    const double tolerance = 1e-12;
    for (int i = 0; i < max_iterations; ++i) {
        sin_lat = std::sin(lat);
        N = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat); // Radius of curvature
        h = p / std::cos(lat) - N; // Altitude
        double new_lat = std::atan2(z, p * (1.0 - e2 * N / (N + h)));
        if (std::abs(new_lat - lat) < tolerance) {
            break;
        }
        lat = new_lat;
    }

    // Convert radians to degrees
    lat = lat * 180.0 / M_PI;
    lon = lon * 180.0 / M_PI;

    // Return LLA coordinates
    Eigen::Vector3d lla;
    lla(0) = lat; // Latitude in degrees
    lla(1) = lon; // Longitude in degrees
    lla(2) = h;   // Altitude in meters
    return lla;
}
