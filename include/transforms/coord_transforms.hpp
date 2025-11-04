#pragma once

/// @file coord_frame_transformations.hpp
/// @brief Defines classes for ECI/ECEF coordinate transformations and EOP parsing.
///
/// Provides the EopEntry struct for storing Earth Orientation Parameters (EOP),
/// the EopParser class for loading and interpolating IERS EOP data, and the
/// CoordTransforms class for ECI-to-ECEF, ECEF-to-ECI, and LLA conversions
/// using the SOFA library.

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace transforms {
/// @struct EopEntry
/// @brief Stores Earth Orientation Parameters (EOP) for a specific time.
///
/// Contains polar motion (x_p, y_p), UT1-UTC difference, and the corresponding
/// Modified Julian Date (MJD).
struct EopEntry {
    double mjd;    ///< Modified Julian Date (days).
    double xp;     ///< Polar motion X coordinate (degrees).
    double yp;     ///< Polar motion Y coordinate (degrees).
    double dut1;   ///< UT1-UTC time difference (seconds).
};

/// @class EopParser
/// @brief Parses and interpolates Earth Orientation Parameters from IERS data files.
///
/// Loads EOP data from files like finals2000A.all and provides interpolation
/// for specific MJD values to support precise ECI/ECEF transformations.
class EopParser {
public:
    /// @brief Constructs an EopParser by loading EOP data from a file.
    /// @param filename Path to the IERS EOP file (e.g., finals2000A.all).
    /// @throws std::runtime_error If the file cannot be opened or contains no valid data.
    EopParser(const std::string& filename);

    /// @brief Interpolates EOP data for a given Modified Julian Date (MJD).
    /// @param mjd The Modified Julian Date to interpolate (days).
    /// @return EopEntry containing interpolated xp, yp, and dut1 values.
    /// @throws std::runtime_error If EOP data is empty or MJD is out of range.
    /// @note Uses linear interpolation; returns edge values for out-of-range MJD.
    auto getEopForMjd(double mjd) const -> EopEntry;

private:
    std::vector<EopEntry> entries_; ///< Stored EOP data entries.
};

/// @class CoordTransforms
/// @brief Handles coordinate transformations between ECI, ECEF, and LLA frames.
///
/// Uses the SOFA library for high-precision transformations between Earth-Centered
/// Inertial (ECI, CIRS) and Earth-Centered Earth-Fixed (ECEF, ITRS) frames, as well
/// as conversions to/from Latitude, Longitude, Altitude (LLA). Incorporates EOP data
/// for accurate rotations. Used in ballistic propagation simulations.
class CoordTransforms {
public:
    /// @brief Constructs a CoordTransforms object, initializing EOP parser.
    /// @param filename Path to the IERS EOP file (e.g., finals2000A.all).
    /// @throws std::runtime_error If EOP file loading fails.
    /// @note Uses IERS_EOP_FILE macro defined by CMake.
    CoordTransforms(const std::string& filename);

    /// @brief Converts a 6D state vector from ECI to ECEF.
    /// @param state_eci 6D state vector [x, y, z, vx, vy, vz] in ECI (meters, m/s).
    /// @param t Time in seconds since J2000 epoch.
    /// @return 6D state vector [x, y, z, vx, vy, vz] in ECEF (meters, m/s).
    /// @throws std::invalid_argument If state_eci is not 6D.
    /// @throws std::runtime_error If EOP interpolation or time conversion fails.
    /// @note Uses SOFA's iauC2t06a for the rotation matrix, incorporating EOP data.
    auto eci_to_ecef(const Eigen::VectorXd& state_eci, double t) const -> Eigen::VectorXd;

    /// @brief Converts a 6D state vector from ECEF to ECI.
    /// @param state_ecef 6D state vector [x, y, z, vx, vy, vz] in ECEF (meters, m/s).
    /// @param t Time in seconds since J2000 epoch.
    /// @return 6D state vector [x, y, z, vx, vy, vz] in ECI (meters, m/s).
    /// @throws std::invalid_argument If state_ecef is not 6D.
    /// @throws std::runtime_error If EOP interpolation or time conversion fails.
    /// @note Uses SOFA's iauC2t06a for the rotation matrix, incorporating EOP data.
    auto ecef_to_eci(const Eigen::VectorXd& state_ecef, double t) const -> Eigen::VectorXd;

    /// @brief Converts geodetic LLA (Latitude, Longitude, Altitude) to ECEF coordinates.
    /// @param lat_deg Latitude in degrees.
    /// @param lon_deg Longitude in degrees.
    /// @param alt_m Altitude above WGS84 ellipsoid in meters.
    /// @return 3D vector [x, y, z] in ECEF coordinates (meters).
    /// @throws std::runtime_error If the conversion fails (e.g., invalid ellipsoid).
    /// @note Uses SOFA's iauGd2gc with WGS84 ellipsoid (n=1).
    auto lla_to_ecef(double lat_deg, double lon_deg, double alt_m) const -> Eigen::VectorXd;

    /// @brief Converts ECEF coordinates to geodetic LLA (Latitude, Longitude, Altitude).
    /// @param ecef 3D vector [x, y, z] in ECEF coordinates (meters).
    /// @return 3D vector [latitude (degrees), longitude (degrees), altitude (meters)].
    /// @throws std::runtime_error If the conversion fails (e.g., invalid coordinates).
    /// @note Uses SOFA's iauGc2gd with WGS84 ellipsoid (n=1).
    auto ecef_to_lla(const Eigen::Vector3d& ecef) const -> Eigen::Vector3d;

private:
    EopParser parser_; ///< EOP parser for loading and interpolating IERS data.
};
} // namespace transforms
