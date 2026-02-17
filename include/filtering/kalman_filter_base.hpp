#pragma once

#include <Eigen/Dense>

#include "common/types.hpp"

namespace filtering {

/// @brief Pure interface defining the contract for all Kalman filters
class IKalmanFilter {
public:
    /// @brief Virtual destructor
    virtual ~IKalmanFilter() = default;

    /// @brief Predict the state forward by a time interval
    /// @param dt Time step for prediction
    virtual void predict(double dt) = 0;

    /// @brief Update the filter with a new measurement
    /// @param measurement Measurement to incorporate into the state estimate
    virtual void update(const common::Measurement& measurement) = 0;

    /// @brief Compute Gaussian likelihood of a measurement
    /// 
    /// Calculates the probability of observing the given measurement
    /// given the current state estimate. Used by IMM for mode probability updates.
    /// 
    /// @param measurement Measurement to evaluate
    /// @return Likelihood value (non-negative, typically in range [0, 1])
    virtual double get_innovation_likelihood(const common::Measurement& measurement) const = 0;
    
    /// @brief Get the current state vector
    /// @return State vector
    virtual Eigen::VectorXd get_state() const = 0;
    
    /// @brief Get the current covariance matrix
    /// @return Covariance matrix
    virtual Eigen::MatrixXd get_covariance() const = 0;
    
    /// @brief Set the filter's state
    /// @param state State vector to set
    virtual void set_state(const Eigen::VectorXd& state) = 0;
    
    /// @brief Set the filter's covariance matrix
    /// @param covariance Covariance matrix to set
    virtual void set_covariance(const Eigen::MatrixXd& covariance) = 0;

    /// @brief Get the current filter time
    /// @return Current filter time in seconds
    virtual double get_time() const = 0;
};

/// @brief Abstract base class providing common functionality
/// 
/// Implements IKalmanFilter and adds helpful methods
class KalmanFilterBase : public IKalmanFilter {
public:
    /// @brief Virtual destructor
    virtual ~KalmanFilterBase() = default;

    // ========================================
    // IKalmanFilter interface - Pure virtual
    // ========================================

    /// @copydoc IKalmanFilter::predict()
    virtual void predict(double dt) = 0;
    /// @copydoc IKalmanFilter::update()
    virtual void update(const common::Measurement& measurement) = 0;
    /// @copydoc IKalmanFilter::get_state()
    virtual Eigen::VectorXd get_state() const = 0;
    /// @copydoc IKalmanFilter::get_covariance()
    virtual Eigen::MatrixXd get_covariance() const = 0;
    /// @copydoc IKalmanFilter::set_state()
    virtual void set_state(const Eigen::VectorXd& state) = 0;
    /// @copydoc IKalmanFilter::set_covariance()
    virtual void set_covariance(const Eigen::MatrixXd& covariance) = 0;
    /// @copydoc IKalmanFilter::get_time()
    virtual double get_time() const = 0;

    // ========================================
    // HELPER METHODS - Concrete implementation
    // ========================================
    
    /// @brief Reset the filter state and covariance
    /// @param initial_state Initial state vector
    /// @param initial_covariance Initial covariance matrix
    virtual void reset(
        const Eigen::VectorXd& initial_state,
        const Eigen::MatrixXd& initial_covariance
    ) {
        set_state(initial_state);
        set_covariance(initial_covariance);
    }

    /// @brief Get the number of dimensions of the state vector
    /// @return State vector dimension
    virtual int get_state_dimension() const {
        return get_state().size();
    }

    /// @brief Get the position component of the state vector
    /// @return Position as a 3D vector
    virtual Eigen::Vector3d get_position() const {
        Eigen::VectorXd state = get_state();
        if (state.size() >= 3) {
            return state.head<3>();
        }
        return Eigen::Vector3d::Zero();
    }

    /// @brief Get the velocity component of the state vector
    /// @return Velocity as a 3D vector
    virtual Eigen::Vector3d get_velocity() const {
        Eigen::VectorXd state = get_state();
        if (state.size() >= 6) {
            return state.segment<3>(3);
        }
        return Eigen::Vector3d::Zero();
    }

    /// @brief Get the position uncertainty from the covariance matrix
    /// @return Position uncertainty as a 3D vector
    virtual Eigen::Vector3d get_position_uncertainty() const {
        Eigen::MatrixXd P = get_covariance();
        if (P.rows() >= 3 && P.cols() >= 3) {
            return P.block<3, 3>(0, 0).diagonal().cwiseSqrt();
        }
        return Eigen::Vector3d::Zero();
    }

    /// @brief Check if the covariance matrix is valid (symmetric and positive semi-definite)
    /// @return True if the covariance matrix is valid, false otherwise
    virtual bool is_covariance_valid() const {
        Eigen::MatrixXd P = get_covariance();
        if (!P.isApprox(P.transpose(), 1e-9)) {
            return false;
        }
        Eigen::LLT<Eigen::MatrixXd> llt(P);
        return llt.info() == Eigen::Success;
    }

    /// @brief Compute the innovation (measurement residual)
    /// @param measurement The actual measurement
    /// @param predicted_measurement The predicted measurement
    /// @return Innovation vector
    virtual Eigen::VectorXd compute_innovation(
        const common::Measurement& measurement,
        const Eigen::VectorXd& predicted_measurement
    ) const {
        return measurement.z - predicted_measurement;
    }

    /// @brief Compute the Normalized Innovation Squared (NIS)
    /// @param innovation Innovation vector
    /// @param innovation_covariance Innovation covariance matrix
    /// @return NIS value
    virtual double compute_nis(
        const Eigen::VectorXd& innovation,
        const Eigen::MatrixXd& innovation_covariance
    ) const {
        return innovation.transpose() * innovation_covariance.inverse() * innovation;
    }
};

} // namespace filtering
