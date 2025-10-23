#pragma once

/// @file coord_frame_transformations.hpp
/// @brief Defines classes for ECI/ECEF coordinate transformations and EOP parsing.
///
/// Provides the EopEntry struct for storing Earth Orientation Parameters (EOP),
/// the EopParser class for loading and interpolating IERS EOP data, and the
/// CoordTransforms class for ECI-to-ECEF, ECEF-to-ECI, and LLA conversions
/// using the SOFA library.

#include <Eigen/Dense>
#include <sofa.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

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
    EopParser(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open EOP file: " + filename);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines
            std::istringstream iss(line);
            double mjd, xp, yp, dut1;
            // finals2000A.all format: Columns 8-9 (x_pole, y_pole in arcsec), 16 (UT1-UTC in sec)
            std::string dummy;
            iss >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> xp >> yp;
            iss >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dut1;
            if (iss.fail()) continue; // Skip malformed lines
            entries_.push_back({mjd, xp / 3600.0, yp / 3600.0, dut1}); // Convert arcsec to degrees
        }
        if (entries_.empty()) {
            throw std::runtime_error("No valid EOP data found in: " + filename);
        }
    }

    /// @brief Interpolates EOP data for a given Modified Julian Date (MJD).
    /// @param mjd The Modified Julian Date to interpolate (days).
    /// @return EopEntry containing interpolated xp, yp, and dut1 values.
    /// @throws std::runtime_error If EOP data is empty or MJD is out of range.
    /// @note Uses linear interpolation; returns edge values for out-of-range MJD.
    EopEntry getEopForMjd(double mjd) const {
        if (entries_.empty()) {
            throw std::runtime_error("EOP entries empty");
        }
        // Handle edge cases: Return first/last entry if mjd is out of range
        if (mjd <= entries_.front().mjd) return entries_.front();
        if (mjd >= entries_.back().mjd) return entries_.back();

        // Linear interpolation
        for (size_t i = 0; i < entries_.size() - 1; ++i) {
            if (entries_[i].mjd <= mjd && mjd <= entries_[i + 1].mjd) {
                double frac = (mjd - entries_[i].mjd) / (entries_[i + 1].mjd - entries_[i].mjd);
                return {
                    mjd,
                    entries_[i].xp + frac * (entries_[i + 1].xp - entries_[i].xp),
                    entries_[i].yp + frac * (entries_[i + 1].yp - entries_[i].yp),
                    entries_[i].dut1 + frac * (entries_[i + 1].dut1 - entries_[i].dut1)
                };
            }
        }
        throw std::runtime_error("MJD out of EOP range");
    }

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
    CoordTransforms(const std::string& filename)
    : parser_(filename)
    {}

    /// @brief Converts a 6D state vector from ECI to ECEF.
    /// @param state_eci 6D state vector [x, y, z, vx, vy, vz] in ECI (meters, m/s).
    /// @param t Time in seconds since J2000 epoch.
    /// @return 6D state vector [x, y, z, vx, vy, vz] in ECEF (meters, m/s).
    /// @throws std::invalid_argument If state_eci is not 6D.
    /// @throws std::runtime_error If EOP interpolation or time conversion fails.
    /// @note Uses SOFA's iauC2t06a for the rotation matrix, incorporating EOP data.
    Eigen::VectorXd eci_to_ecef(const Eigen::VectorXd& state_eci, double t) const {
        if (state_eci.size() != 6) {
            throw std::invalid_argument("State vector must have 6 elements");
        }

        Eigen::Vector3d r_eci = state_eci.segment<3>(0); // Position [x, y, z]
        Eigen::Vector3d v_eci = state_eci.segment<3>(3); // Velocity [vx, vy, vz]

        // Convert time (seconds since J2000) to two-part Julian Date
        double jd_utc1 = 2451545.0; // J2000 epoch
        double jd_utc2 = t / 86400.0; // Seconds to days
        double mjd = jd_utc1 + jd_utc2 - 2400000.5;

        // Interpolate EOP for the given MJD
        EopEntry eop;
        try {
            eop = parser_.getEopForMjd(mjd);
        } catch (const std::exception& e) {
            throw std::runtime_error("EOP interpolation failed: " + std::string(e.what()));
        }

        // Convert UTC to TAI and TT
        double jd_tai1, jd_tai2, jd_tt1, jd_tt2;
        int status = iauUtctai(jd_utc1, jd_utc2, &jd_tai1, &jd_tai2);
        if (status != 0) {
            throw std::runtime_error("UTC to TAI conversion failed with status: " + std::to_string(status));
        }
        status = iauTaitt(jd_tai1, jd_tai2, &jd_tt1, &jd_tt2);
        if (status != 0) {
            throw std::runtime_error("TAI to TT conversion failed with status: " + std::to_string(status));
        }

        // Convert UTC to UT1 using dut1
        double jd_ut1_1 = jd_utc1;
        double jd_ut1_2 = jd_utc2 + eop.dut1 / 86400.0; // Add UT1-UTC difference in days

        // Compute ECI to ECEF rotation matrix
        double r[3][3];
        iauC2t06a(jd_tt1, jd_tt2, jd_ut1_1, jd_ut1_2, eop.xp, eop.yp, r);

        // Populate rotation matrix
        Eigen::Matrix3d R;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R(i, j) = r[i][j];
            }
        }

        // Transform position
        Eigen::Vector3d r_ecef = R * r_eci;

        // Transform velocity, accounting for Earth's rotation
        Eigen::Vector3d omega(0.0, 0.0, 7.292115e-5); // Earth's angular velocity (rad/s)
        Eigen::Vector3d v_ecef = R * v_eci - omega.cross(r_ecef);

        Eigen::VectorXd state_ecef(6);
        state_ecef << r_ecef, v_ecef;
        return state_ecef;
    }

    /// @brief Converts a 6D state vector from ECEF to ECI.
    /// @param state_ecef 6D state vector [x, y, z, vx, vy, vz] in ECEF (meters, m/s).
    /// @param t Time in seconds since J2000 epoch.
    /// @return 6D state vector [x, y, z, vx, vy, vz] in ECI (meters, m/s).
    /// @throws std::invalid_argument If state_ecef is not 6D.
    /// @throws std::runtime_error If EOP interpolation or time conversion fails.
    /// @note Uses SOFA's iauC2t06a for the rotation matrix, incorporating EOP data.
    Eigen::VectorXd ecef_to_eci(const Eigen::VectorXd& state_ecef, double t) const {
        if (state_ecef.size() != 6) {
            throw std::invalid_argument("State vector must have 6 elements");
        }

        Eigen::Vector3d r_ecef = state_ecef.segment<3>(0); // Position [x, y, z]
        Eigen::Vector3d v_ecef = state_ecef.segment<3>(3); // Velocity [vx, vy, vz]

        // Convert time (seconds since J2000) to two-part Julian Date
        double jd_utc1 = 2451545.0; // J2000 epoch
        double jd_utc2 = t / 86400.0; // Seconds to days
        double mjd = jd_utc1 + jd_utc2 - 2400000.5;

        // Interpolate EOP for the given MJD
        EopEntry eop;
        try {
            eop = parser_.getEopForMjd(mjd);
        } catch (const std::exception& e) {
            throw std::runtime_error("EOP interpolation failed: " + std::string(e.what()));
        }

        // Convert UTC to TAI and TT
        double jd_tai1, jd_tai2, jd_tt1, jd_tt2;
        int status = iauUtctai(jd_utc1, jd_utc2, &jd_tai1, &jd_tai2);
        if (status != 0) {
            throw std::runtime_error("UTC to TAI conversion failed with status: " + std::to_string(status));
        }
        status = iauTaitt(jd_tai1, jd_tai2, &jd_tt1, &jd_tt2);
        if (status != 0) {
            throw std::runtime_error("TAI to TT conversion failed with status: " + std::to_string(status));
        }

        // Convert UTC to UT1 using dut1
        double jd_ut1_1 = jd_utc1;
        double jd_ut1_2 = jd_utc2 + eop.dut1 / 86400.0; // Add UT1-UTC difference in days

        // Compute ECI to ECEF rotation matrix
        double r[3][3];
        iauC2t06a(jd_tt1, jd_tt2, jd_ut1_1, jd_ut1_2, eop.xp, eop.yp, r);

        // Invert for ECEF to ECI (transpose, as rotation matrix is orthogonal)
        Eigen::Matrix3d R;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R(i, j) = r[j][i]; // Transpose
            }
        }

        // Transform position
        Eigen::Vector3d r_eci = R * r_ecef;

        // Transform velocity, accounting for Earth's rotation
        Eigen::Vector3d omega(0.0, 0.0, 7.292115e-5); // Earth's angular velocity (rad/s)
        Eigen::Vector3d v_eci = R * (v_ecef + omega.cross(r_ecef));

        Eigen::VectorXd state_eci(6);
        state_eci << r_eci, v_eci;
        return state_eci;
    }

    /// @brief Converts geodetic LLA (Latitude, Longitude, Altitude) to ECEF coordinates.
    /// @param lat_deg Latitude in degrees.
    /// @param lon_deg Longitude in degrees.
    /// @param alt_m Altitude above WGS84 ellipsoid in meters.
    /// @return 3D vector [x, y, z] in ECEF coordinates (meters).
    /// @throws std::runtime_error If the conversion fails (e.g., invalid ellipsoid).
    /// @note Uses SOFA's iauGd2gc with WGS84 ellipsoid (n=1).
    Eigen::VectorXd lla_to_ecef(double lat_deg, double lon_deg, double alt_m) const {
        // Convert degrees to radians
        double lat_rad = lat_deg * M_PI / 180.0;
        double lon_rad = lon_deg * M_PI / 180.0;

        // SOFA: Geodetic to geocentric Cartesian (WGS84 ellipsoid, n=1)
        double xyz[3];
        int status = iauGd2gc(1, lon_rad, lat_rad, alt_m, xyz); // 1 = WGS84
        if (status != 0) {
            throw std::runtime_error("LLA to ECEF conversion failed with status: " + std::to_string(status));
        }

        return Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
    }

    /// @brief Converts ECEF coordinates to geodetic LLA (Latitude, Longitude, Altitude).
    /// @param ecef 3D vector [x, y, z] in ECEF coordinates (meters).
    /// @return 3D vector [latitude (degrees), longitude (degrees), altitude (meters)].
    /// @throws std::runtime_error If the conversion fails (e.g., invalid coordinates).
    /// @note Uses SOFA's iauGc2gd with WGS84 ellipsoid (n=1).
    Eigen::Vector3d ecef_to_lla(const Eigen::Vector3d& ecef) const {
        double xyz[3] = {ecef(0), ecef(1), ecef(2)};
        double lon, lat, alt;
        int status = iauGc2gd(1, xyz, &lon, &lat, &alt); // 1 = WGS84
        if (status != 0) {
            throw std::runtime_error("ECEF to LLA conversion failed with status: " + std::to_string(status));
        }

        // Convert radians to degrees
        return Eigen::Vector3d(lat * 180.0 / M_PI, lon * 180.0 / M_PI, alt);
    }

private:
    EopParser parser_; ///< EOP parser for loading and interpolating IERS data.
};
