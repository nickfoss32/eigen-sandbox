#include "common/types.hpp"

#include <stdexcept>

namespace common {

// ============================================
// Measurement Implementation
// ============================================

Measurement::Measurement()
    : time(0.0),
      sensor_id(""),
      measurement_id(-1),
      is_associated(false),
      track_id(-1)
{
    // Empty measurement (invalid)
}

Measurement::Measurement(
    const Eigen::VectorXd& z_,
    const Eigen::MatrixXd& R_,
    double time_
) : z(z_),
    R(R_),
    time(time_),
    sensor_id(""),
    measurement_id(-1),
    is_associated(false),
    track_id(-1)
{
    // Validate dimensions
    if (z.size() == 0) {
        throw std::invalid_argument("Measurement vector cannot be empty");
    }
    
    if (R.rows() != z.size() || R.cols() != z.size()) {
        throw std::invalid_argument(
            "Measurement covariance dimensions must match measurement vector dimension"
        );
    }
    
    if (time_ < 0.0) {
        throw std::invalid_argument("Measurement time cannot be negative");
    }
    
    // Check that covariance is symmetric
    if (!R.isApprox(R.transpose(), 1e-10)) {
        throw std::invalid_argument("Measurement covariance must be symmetric");
    }
    
    // Check that covariance is positive definite (all eigenvalues > 0)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(R);
    if (es.eigenvalues().minCoeff() <= 0.0) {
        throw std::invalid_argument("Measurement covariance must be positive definite");
    }
}

bool Measurement::is_valid() const {
    // Check basic validity
    if (z.size() == 0) {
        return false;
    }
    
    if (R.rows() != z.size() || R.cols() != z.size()) {
        return false;
    }
    
    if (time < 0.0) {
        return false;
    }
    
    // Check for NaN or Inf in measurement vector
    if (!z.allFinite()) {
        return false;
    }
    
    // Check for NaN or Inf in covariance
    if (!R.allFinite()) {
        return false;
    }
    
    // Check symmetry
    if (!R.isApprox(R.transpose(), 1e-10)) {
        return false;
    }
    
    // Check positive definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(R);
    if (es.eigenvalues().minCoeff() <= 0.0) {
        return false;
    }
    
    return true;
}

} // namespace common