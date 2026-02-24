#pragma once

#include "common/types.hpp"

#include <Eigen/Dense>

#include <string>

namespace estimation {

/**
 * @brief Interface for state estimation algorithms
 * 
 * This interface defines the contract for all state estimators (EKF, UKF, IMM, etc.)
 * Used by Track objects to estimate target state from measurements.
 */
class IStateEstimator {
public:
    virtual ~IStateEstimator() = default;
    
    /**
     * @brief Predict state forward in time
     * @param dt Time step in seconds
     */
    virtual void predict(double dt) = 0;
    
    /**
     * @brief Update state with a new measurement
     * @param measurement Measurement to incorporate
     */
    virtual void update(const common::Measurement& measurement) = 0;
    
    /**
     * @brief Get current state estimate
     * @return State vector
     */
    virtual Eigen::VectorXd get_state() const = 0;
    
    /**
     * @brief Get current state covariance
     * @return Covariance matrix
     */
    virtual Eigen::MatrixXd get_covariance() const = 0;
    
    /**
     * @brief Set state estimate
     * @param state New state vector
     */
    virtual void set_state(const Eigen::VectorXd& state) = 0;
    
    /**
     * @brief Set state covariance
     * @param covariance New covariance matrix
     */
    virtual void set_covariance(const Eigen::MatrixXd& covariance) = 0;
    
    /**
     * @brief Get current time of the estimate
     * @return Time in seconds
     */
    virtual double get_time() const = 0;
    
    /**
     * @brief Get estimator type identifier
     * @return String describing estimator type (e.g., "IMM", "EKF")
     */
    virtual std::string get_type() const = 0;
    
    /**
     * @brief Get state dimension
     * @return State vector size
     */
    virtual int get_state_dimension() const = 0;
};
} // namespace estimation
