#include "estimation/imm.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace estimation {

/// @brief Tolerance for checking time consistency between filters
constexpr double kTimeConsistencyTolerance = 1e-9;

// Multi-model constructor
IMM::IMM(
    std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters,
    const Eigen::VectorXd& initial_model_probs,
    const Eigen::MatrixXd& transition_matrix
)
    : filters_(std::move(filters))
    , model_probs_(initial_model_probs)
    , transition_matrix_(transition_matrix)
    , num_models_(static_cast<int>(filters_.size()))
    , state_dim_(0)
{
    if (filters_.empty()) {
        throw std::invalid_argument("IMM: filters vector cannot be empty");
    }
    if (!filters_[0]) {
        throw std::invalid_argument("IMM: filters cannot contain null pointers");
    }

    state_dim_ = static_cast<int>(filters_[0]->get_state().size());

    validate_inputs();
    
    // Initialize intermediate variables
    mixing_probs_.resize(num_models_, num_models_);
    model_probs_pred_.resize(num_models_);
    mixed_states_.resize(num_models_);
    mixed_covariances_.resize(num_models_);
    likelihoods_.resize(num_models_);
    
    for (int i = 0; i < num_models_; ++i) {
        mixed_states_[i].resize(state_dim_);
        mixed_covariances_[i].resize(state_dim_, state_dim_);
    }
    
    combined_state_.resize(state_dim_);
    combined_covariance_.resize(state_dim_, state_dim_);
    
    // Initialize combined estimate
    combination();
}

// Single-model constructor
IMM::IMM(std::unique_ptr<filtering::IKalmanFilter> filter)
    : IMM(
        [&filter]() {
            std::vector<std::unique_ptr<filtering::IKalmanFilter>> vec;
            vec.push_back(std::move(filter));
            return vec;
        }(),
        Eigen::VectorXd::Ones(1),
        Eigen::MatrixXd::Ones(1, 1)
    )
{}

void IMM::validate_inputs() {
    if (model_probs_.size() != num_models_) {
        throw std::invalid_argument("IMM: model probabilities size must match number of models");
    }
    
    if (transition_matrix_.rows() != num_models_ || transition_matrix_.cols() != num_models_) {
        throw std::invalid_argument("IMM: transition matrix must be square with size = number of models");
    }
    
    // Check model probabilities are valid and sum to 1.
    for (int i = 0; i < num_models_; ++i) {
        if (!std::isfinite(model_probs_(i)) || model_probs_(i) < 0.0) {
            throw std::invalid_argument("IMM: model probabilities must be finite and non-negative");
        }
    }

    double prob_sum = model_probs_.sum();
    if (std::abs(prob_sum - 1.0) > 1e-6) {
        throw std::invalid_argument("IMM: initial mode probabilities must sum to 1.0");
    }
    
    // Check transition matrix is row-stochastic with non-negative entries.
    for (int i = 0; i < num_models_; ++i) {
        for (int j = 0; j < num_models_; ++j) {
            if (!std::isfinite(transition_matrix_(i, j)) || transition_matrix_(i, j) < 0.0) {
                throw std::invalid_argument("IMM: transition matrix entries must be finite and non-negative");
            }
        }

        double row_sum = transition_matrix_.row(i).sum();
        if (std::abs(row_sum - 1.0) > 1e-6) {
            throw std::invalid_argument("IMM: each row of transition matrix must sum to 1.0");
        }
    }
    
    // Check all filters are valid and have matching dimensions/time.
    const double reference_time = filters_[0]->get_time();
    if (!std::isfinite(reference_time)) {
        throw std::invalid_argument("IMM: filter times must be finite");
    }

    for (const auto& filter : filters_) {
        if (!filter) {
            throw std::invalid_argument("IMM: filters cannot contain null pointers");
        }

        if (filter->get_state().size() != state_dim_) {
            throw std::invalid_argument("IMM: all filters must have same state dimension");
        }

        if (filter->get_covariance().rows() != state_dim_ ||
            filter->get_covariance().cols() != state_dim_) {
            throw std::invalid_argument("IMM: all filters must have covariance size = state dimension");
        }

        const double filter_time = filter->get_time();
        if (!std::isfinite(filter_time)) {
            throw std::invalid_argument("IMM: filter times must be finite");
        }
        if (std::abs(filter_time - reference_time) > kTimeConsistencyTolerance) {
            throw std::invalid_argument("IMM: all filters must have consistent time");
        }
    }
}

void IMM::predict(double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("IMM::predict: dt must be positive");
    }

    if (is_single_model()) {
        // Single-model case: direct prediction
        filters_[0]->predict(dt);
        validate_time_consistency();
        combined_state_ = filters_[0]->get_state();
        combined_covariance_ = filters_[0]->get_covariance();
        model_probs_pred_ = model_probs_;
        return;
    }
    
    // Full IMM prediction cycle
    mixing();
    model_matched_filtering(dt);
    validate_time_consistency();
    model_probs_ = model_probs_pred_;
    combination();
}

void IMM::update(const common::Measurement& measurement) {
    if (is_single_model()) {
        // Single-model case: direct update
        filters_[0]->update(measurement);
        validate_time_consistency();
        combined_state_ = filters_[0]->get_state();
        combined_covariance_ = filters_[0]->get_covariance();
        return;
    }

    // If update() is called before predict(), fall back to current model probabilities.
    if (model_probs_pred_.size() != num_models_) {
        model_probs_pred_ = model_probs_;
    }

    // Update model probabilities based on pre-update innovation likelihoods.
    model_probability_update(measurement);

    // Run model-conditioned measurement updates.
    for (auto& filter : filters_) {
        filter->update(measurement);
    }
    validate_time_consistency();

    // Combine updated estimates.
    combination();
}

void IMM::mixing() {
    // Compute predicted model probabilities: μ_j(k|k-1) = Σ_i p_ij * μ_i(k-1)
    model_probs_pred_ = transition_matrix_.transpose() * model_probs_;
    
    // Compute mixing probabilities: μ_i|j(k-1) = p_ij * μ_i(k-1) / μ_j(k|k-1)
    for (int j = 0; j < num_models_; ++j) {
        for (int i = 0; i < num_models_; ++i) {
            if (model_probs_pred_(j) > 1e-10) {
                mixing_probs_(i, j) = transition_matrix_(i, j) * model_probs_(i) / model_probs_pred_(j);
            } else {
                mixing_probs_(i, j) = 1.0 / num_models_;  // Uniform if denominator is zero
            }
        }
    }
    
    // Compute mixed initial conditions for each filter
    for (int j = 0; j < num_models_; ++j) {
        // Mixed state: x̂_0j(k-1|k-1) = Σ_i x̂_i(k-1|k-1) * μ_i|j(k-1)
        mixed_states_[j].setZero();
        for (int i = 0; i < num_models_; ++i) {
            mixed_states_[j] += mixing_probs_(i, j) * filters_[i]->get_state();
        }
        
        // Mixed covariance: P_0j(k-1|k-1) = Σ_i μ_i|j * [P_i + (x̂_i - x̂_0j)(x̂_i - x̂_0j)^T]
        mixed_covariances_[j].setZero();
        for (int i = 0; i < num_models_; ++i) {
            Eigen::VectorXd state_diff = filters_[i]->get_state() - mixed_states_[j];
            mixed_covariances_[j] += mixing_probs_(i, j) * 
                (filters_[i]->get_covariance() + state_diff * state_diff.transpose());
        }
    }
}

void IMM::model_matched_filtering(double dt) {
    // Set mixed initial conditions and predict each filter
    for (int j = 0; j < num_models_; ++j) {
        filters_[j]->set_state(mixed_states_[j]);
        filters_[j]->set_covariance(mixed_covariances_[j]);
        filters_[j]->predict(dt);
    }
}

void IMM::model_probability_update(const common::Measurement& measurement) {
    // Compute likelihood for each filter
    for (int j = 0; j < num_models_; ++j) {
        likelihoods_[j] = compute_likelihood(j, measurement);
    }
    
    // Update mode probabilities: μ_j(k) = Λ_j(k) * μ_j(k|k-1) / c
    // where c = Σ_j Λ_j(k) * μ_j(k|k-1) is the normalization constant
    double normalization = 0.0;
    for (int j = 0; j < num_models_; ++j) {
        normalization += likelihoods_[j] * model_probs_pred_(j);
    }
    
    if (normalization > 1e-10) {
        for (int j = 0; j < num_models_; ++j) {
            model_probs_(j) = likelihoods_[j] * model_probs_pred_(j) / normalization;
        }
    } else {
        // If normalization is too small, reset to uniform
        model_probs_.setConstant(1.0 / num_models_);
    }
}

void IMM::combination() {
    // Combined state: x̂(k|k) = Σ_j x̂_j(k|k) * μ_j(k)
    combined_state_.setZero();
    for (int j = 0; j < num_models_; ++j) {
        combined_state_ += model_probs_(j) * filters_[j]->get_state();
    }
    
    // Combined covariance: P(k|k) = Σ_j μ_j * [P_j + (x̂_j - x̂)(x̂_j - x̂)^T]
    combined_covariance_.setZero();
    for (int j = 0; j < num_models_; ++j) {
        Eigen::VectorXd state_diff = filters_[j]->get_state() - combined_state_;
        combined_covariance_ += model_probs_(j) * 
            (filters_[j]->get_covariance() + state_diff * state_diff.transpose());
    }
}

double IMM::compute_likelihood(int filter_idx, const common::Measurement& measurement) {
    constexpr double kMinLikelihood = 1e-12;
    const double likelihood = filters_[filter_idx]->get_innovation_likelihood(measurement);

    if (!std::isfinite(likelihood) || likelihood < 0.0) {
        return kMinLikelihood;
    }

    return std::max(likelihood, kMinLikelihood);
}

Eigen::VectorXd IMM::get_state() const {
    return combined_state_;
}

Eigen::MatrixXd IMM::get_covariance() const {
    return combined_covariance_;
}

void IMM::set_state(const Eigen::VectorXd& state) {
    if (state.size() != state_dim_) {
        throw std::invalid_argument("IMM::set_state: state dimension mismatch");
    }
    
    combined_state_ = state;
    
    // Set state for all filters
    for (auto& filter : filters_) {
        filter->set_state(state);
    }
}

void IMM::set_covariance(const Eigen::MatrixXd& covariance) {
    if (covariance.rows() != state_dim_ || covariance.cols() != state_dim_) {
        throw std::invalid_argument("IMM::set_covariance: covariance dimension mismatch");
    }
    
    combined_covariance_ = covariance;
    
    // Set covariance for all filters
    for (auto& filter : filters_) {
        filter->set_covariance(covariance);
    }
}

double IMM::get_time() const {
    validate_time_consistency();
    return filters_[0]->get_time();
}

void IMM::validate_time_consistency() const {
    if (filters_.empty() || !filters_[0]) {
        throw std::runtime_error("IMM: cannot validate time consistency with empty filters");
    }

    const double reference_time = filters_[0]->get_time();
    if (!std::isfinite(reference_time)) {
        throw std::runtime_error("IMM: filter times must be finite");
    }

    for (int i = 1; i < num_models_; ++i) {
        if (!filters_[i]) {
            throw std::runtime_error("IMM: filters cannot contain null pointers");
        }

        const double filter_time = filters_[i]->get_time();
        if (!std::isfinite(filter_time)) {
            throw std::runtime_error("IMM: filter times must be finite");
        }
        if (std::abs(filter_time - reference_time) > kTimeConsistencyTolerance) {
            throw std::runtime_error("IMM: filter times are inconsistent");
        }
    }
}

std::string IMM::get_type() const {
    std::ostringstream oss;
    oss << "IMM(" << num_models_ << " model" << (num_models_ > 1 ? "s" : "") << ")";
    return oss.str();
}

int IMM::get_state_dimension() const {
    return state_dim_;
}

int IMM::get_most_likely_model() const {
    int max_idx = 0;
    double max_prob = model_probs_(0);
    
    for (int i = 1; i < num_models_; ++i) {
        if (model_probs_(i) > max_prob) {
            max_prob = model_probs_(i);
            max_idx = i;
        }
    }
    
    return max_idx;
}

Eigen::VectorXd IMM::get_model_state(int model_idx) const {
    if (model_idx < 0 || model_idx >= num_models_) {
        throw std::out_of_range("IMM::get_model_state: invalid mode index");
    }
    return filters_[model_idx]->get_state();
}

Eigen::MatrixXd IMM::get_model_covariance(int model_idx) const {
    if (model_idx < 0 || model_idx >= num_models_) {
        throw std::out_of_range("IMM::get_model_covariance: invalid mode index");
    }
    return filters_[model_idx]->get_covariance();
}

} // namespace estimation
