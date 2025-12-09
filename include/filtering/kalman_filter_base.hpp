#pragma once

#include <Eigen/Dense>

#include "common/types.hpp"

namespace filtering {

/// @brief Pure interface defining the contract for all Kalman filters
class IKalmanFilter {
public:
    virtual ~IKalmanFilter() = default;

    // Core operations - MUST implement
    virtual void predict(double dt) = 0;
    virtual void update(const common::Measurement& measurement) = 0;
    
    // State access - MUST implement
    virtual Eigen::VectorXd get_state() const = 0;
    virtual Eigen::MatrixXd get_covariance() const = 0;
    virtual void set_state(const Eigen::VectorXd& state) = 0;
    virtual void set_covariance(const Eigen::MatrixXd& covariance) = 0;
    virtual double get_time() const = 0;
};

/// @brief Abstract base class providing common functionality
/// 
/// Implements IKalmanFilter and adds helpful methods
class KalmanFilterBase : public IKalmanFilter {
public:
    virtual ~KalmanFilterBase() = default;

    // IKalmanFilter interface - still pure virtual
    virtual void predict(double dt) = 0;
    virtual void update(const common::Measurement& measurement) = 0;
    virtual Eigen::VectorXd get_state() const = 0;
    virtual Eigen::MatrixXd get_covariance() const = 0;
    virtual void set_state(const Eigen::VectorXd& state) = 0;
    virtual void set_covariance(const Eigen::MatrixXd& covariance) = 0;
    virtual double get_time() const = 0;

    // ========================================
    // HELPER METHODS - Concrete implementation
    // ========================================
    
    virtual void reset(
        const Eigen::VectorXd& initial_state,
        const Eigen::MatrixXd& initial_covariance,
        double initial_time = 0.0
    ) {
        set_state(initial_state);
        set_covariance(initial_covariance);
    }

    virtual int get_state_dimension() const {
        return get_state().size();
    }

    virtual Eigen::Vector3d get_position() const {
        Eigen::VectorXd state = get_state();
        if (state.size() >= 3) {
            return state.head<3>();
        }
        return Eigen::Vector3d::Zero();
    }

    virtual Eigen::Vector3d get_velocity() const {
        Eigen::VectorXd state = get_state();
        if (state.size() >= 6) {
            return state.segment<3>(3);
        }
        return Eigen::Vector3d::Zero();
    }

    virtual Eigen::Vector3d get_position_uncertainty() const {
        Eigen::MatrixXd P = get_covariance();
        if (P.rows() >= 3 && P.cols() >= 3) {
            return P.block<3, 3>(0, 0).diagonal().cwiseSqrt();
        }
        return Eigen::Vector3d::Zero();
    }

    virtual bool is_covariance_valid() const {
        Eigen::MatrixXd P = get_covariance();
        if (!P.isApprox(P.transpose(), 1e-9)) {
            return false;
        }
        Eigen::LLT<Eigen::MatrixXd> llt(P);
        return llt.info() == Eigen::Success;
    }

    virtual Eigen::VectorXd compute_innovation(
        const common::Measurement& measurement,
        const Eigen::VectorXd& predicted_measurement
    ) const {
        return measurement.z - predicted_measurement;
    }

    virtual double compute_nis(
        const Eigen::VectorXd& innovation,
        const Eigen::MatrixXd& innovation_covariance
    ) const {
        return innovation.transpose() * innovation_covariance.inverse() * innovation;
    }
};

} // namespace filtering
