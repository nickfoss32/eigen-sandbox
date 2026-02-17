#pragma once

#include "estimation/state_estimator.hpp"
#include "filtering/kalman_filter_base.hpp"
#include "dynamics/dynamics.hpp"
#include "sensor/sensor_model.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <stdexcept>

namespace estimation {


/// @class IMM
/// @implements IStateEstimator
/// @brief Interacting Multiple Model (IMM) estimator
/// 
/// Implements the IMM algorithm for state estimation with multiple dynamic models.
/// Can degenerate to a single-model estimator when constructed with one model.
/// 
/// The IMM cycle consists of:
/// 1. Mixing - Compute mixed initial conditions for each filter
/// 2. Model-matched filtering - Run each filter with its model
/// 3. Model probability update - Update model probabilities based on likelihoods
/// 4. Combination - Combine estimates from all filters
class IMM : public IStateEstimator {
public:
    /// @brief Construct multi-model IMM estimator
    /// @param filters Vector of Kalman filters (one per model)
    /// @param initial_model_probs Initial model probabilities (must sum to 1.0)
    /// @param transition_matrix Markov chain transition matrix (must be stochastic)
    IMM(std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters,
        const Eigen::VectorXd& initial_model_probs,
        const Eigen::MatrixXd& transition_matrix);
    
    /// @brief Construct single-model IMM estimator (degenerates to single filter)
    /// @param filter Single Kalman filter
    explicit IMM(std::unique_ptr<filtering::IKalmanFilter> filter);
    
    ////    IStateEstimator interface implementation    ////
    
    /// @copydoc IStateEstimator::predict()
    void predict(double dt) override;
    /// @copydoc IStateEstimator::update()
    void update(const common::Measurement& measurement) override;
    /// @copydoc IStateEstimator::get_state()
    Eigen::VectorXd get_state() const override;
    /// @copydoc IStateEstimator::get_covariance()
    Eigen::MatrixXd get_covariance() const override;
    /// @copydoc IStateEstimator::set_state()
    void set_state(const Eigen::VectorXd& state) override;
    /// @copydoc IStateEstimator::set_covariance()
    void set_covariance(const Eigen::MatrixXd& covariance) override;
    /// @copydoc IStateEstimator::get_time()
    double get_time() const override;
    /// @copydoc IStateEstimator::get_type()
    std::string get_type() const override;
    /// @copydoc IStateEstimator::get_state_dimension()
    int get_state_dimension() const override;
    
    // IMM-specific methods
    
    /// @brief Get number of models in the IMM
    /// @return Number of models
    int get_num_models() const { return num_models_; }
    
    /// @brief Get current model probabilities
    /// @return Vector of model probabilities (sums to 1.0)
    Eigen::VectorXd get_model_probabilities() const { return model_probs_; }
    
    /// @brief Get index of most likely model
    /// @return Model index with highest probability
    int get_most_likely_model() const;
    
    /// @brief Get state estimate from specific model
    /// @param model_idx Model index
    /// @return State vector from that model's filter
    Eigen::VectorXd get_model_state(int model_idx) const;
    
    /// @brief Get covariance from specific model
    /// @param model_idx Model index
    /// @return Covariance matrix from that model's filter
    Eigen::MatrixXd get_model_covariance(int model_idx) const;
    
private:
    /// @brief Check if this is a single-model estimator
    /// @return true if only one model
    bool is_single_model() const { return num_models_ == 1; }
    
    /// @brief Validate inputs during construction
    void validate_inputs();
    
    /// @brief Step 1: Compute mixing probabilities and mixed initial conditions
    void mixing();
    
    /// @brief Step 2: Run model-matched filtering (predict for each filter)
    /// @param dt Time step
    void model_matched_filtering(double dt);
    
    /// @brief Step 3: Update model probabilities based on filter likelihoods
    /// @param measurement Measurement used for update
    void model_probability_update(const common::Measurement& measurement);
    
    /// @brief Step 4: Combine filter estimates into overall estimate
    void combination();
    
    /// @brief Compute likelihood of measurement for a given filter
    /// @param filter_idx Filter index
    /// @param measurement Measurement
    /// @return Likelihood value
    double compute_likelihood(int filter_idx, const common::Measurement& measurement);

    /// @brief Ensure all model filters are synchronized in time
    void validate_time_consistency() const;
    
    // Member variables
    std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters_;
    
    Eigen::VectorXd model_probs_;         // Current model probabilities μ_i(k)
    Eigen::MatrixXd transition_matrix_;   // Model transition matrix p_ij
    
    int num_models_;
    int state_dim_;
    
    // Intermediate IMM variables
    Eigen::MatrixXd mixing_probs_;        // μ_i|j(k-1) - mixing probabilities
    Eigen::VectorXd model_probs_pred_;    // μ_j(k|k-1) - predicted model probabilities
    std::vector<Eigen::VectorXd> mixed_states_;        // x̂_0j(k-1|k-1)
    std::vector<Eigen::MatrixXd> mixed_covariances_;   // P_0j(k-1|k-1)
    std::vector<double> likelihoods_;     // Λ_j(k) - model likelihoods
    
    // Combined estimate
    Eigen::VectorXd combined_state_;
    Eigen::MatrixXd combined_covariance_;
};

} // namespace estimation
