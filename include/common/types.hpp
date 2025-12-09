#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>
#include <memory>

namespace common {

// ============================================
// MEASUREMENT TYPES (Data Containers)
// ============================================

/// @brief Measurement from a sensor
///
/// Container for actual sensor readings with metadata.
/// Used throughout pipeline: sensors → trackers → filters
struct Measurement {
    Eigen::VectorXd z;              ///< Measurement vector
    Eigen::MatrixXd R;              ///< Measurement noise covariance
    double time;                    ///< Measurement timestamp (seconds)
    std::string sensor_id;          ///< Sensor identifier
    int measurement_id;             ///< Unique measurement ID
    
    // Tracking metadata
    bool is_associated;             ///< Associated to track?
    int track_id;                   ///< Associated track ID (-1 if none)
    
    Measurement();
    Measurement(const Eigen::VectorXd& z_, const Eigen::MatrixXd& R_, double time_);
    
    int dimension() const { return z.size(); }
    bool is_valid() const;
};

/// @brief Batch of measurements
using MeasurementBatch = std::vector<Measurement>;

/// @brief Measurement with source information
struct MeasurementReport {
    Measurement measurement;
    std::string sensor_type;        ///< e.g., "radar", "camera"
    Eigen::Vector3d sensor_position;
    double detection_probability;
    
    MeasurementReport() : detection_probability(1.0) {}
};

// ============================================
// STATE TYPES (Data Containers)
// ============================================

/// @brief State estimate with uncertainty
struct StateEstimate {
    Eigen::VectorXd x;              ///< State vector
    Eigen::MatrixXd P;              ///< State covariance
    double time;                    ///< Estimate timestamp
    
    StateEstimate() : time(0.0) {}
    StateEstimate(const Eigen::VectorXd& x_, const Eigen::MatrixXd& P_, double time_)
        : x(x_), P(P_), time(time_) {}
};

/// @brief Trajectory (sequence of states)
using Trajectory = std::vector<std::pair<double, Eigen::VectorXd>>;

// ============================================
// TRACKING TYPES (Future)
// ============================================

/// @brief Track quality enumeration
enum class TrackQuality {
    TENTATIVE,    ///< New track, not confirmed
    CONFIRMED,    ///< Confirmed track
    COASTING,     ///< No recent measurements
    TERMINATED    ///< Track ended
};

/// @brief Target type enumeration
enum class TargetType {
    UNKNOWN,
    AIRCRAFT,
    MISSILE,
    SATELLITE,
    DEBRIS
};

} // namespace common
