#include "filtering/unscented_transform.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace filtering {

namespace {

constexpr int kMaxJitterAttempts = 8;
constexpr double kBaseJitter = 1e-12;

} // namespace

auto UnscentedTransform::validate_parameters(
    int state_dim,
    const UnscentedTransformParameters& params
) -> double {
    if (state_dim <= 0) {
        throw std::invalid_argument("UnscentedTransform: state dimension must be positive");
    }
    if (!(params.alpha > 0.0)) {
        throw std::invalid_argument("UnscentedTransform: alpha must be positive");
    }
    if (params.beta < 0.0) {
        throw std::invalid_argument("UnscentedTransform: beta must be non-negative");
    }

    const double n = static_cast<double>(state_dim);
    const double lambda = params.alpha * params.alpha * (n + params.kappa) - n;
    if ((n + lambda) <= 0.0) {
        throw std::invalid_argument(
            "UnscentedTransform: n + lambda must be positive (check alpha/kappa)"
        );
    }

    return lambda;
}

auto UnscentedTransform::compute_weights(
    int state_dim,
    const UnscentedTransformParameters& params
) -> Weights {
    const double lambda = validate_parameters(state_dim, params);
    const int sigma_count = 2 * state_dim + 1;
    const double scaling = static_cast<double>(state_dim) + lambda;

    Weights weights;
    weights.lambda = lambda;
    weights.mean = Eigen::VectorXd::Constant(sigma_count, 0.5 / scaling);
    weights.covariance = Eigen::VectorXd::Constant(sigma_count, 0.5 / scaling);

    weights.mean(0) = lambda / scaling;
    weights.covariance(0) =
        weights.mean(0) + (1.0 - params.alpha * params.alpha + params.beta);

    return weights;
}

auto UnscentedTransform::compute_cholesky_factor(const Eigen::MatrixXd& covariance)
    -> Eigen::MatrixXd {
    if (covariance.rows() != covariance.cols()) {
        throw std::invalid_argument(
            "UnscentedTransform: covariance must be square"
        );
    }
    const Eigen::MatrixXd sym_cov = 0.5 * (covariance + covariance.transpose());

    const double diag_scale =
        std::max(1.0, sym_cov.diagonal().cwiseAbs().maxCoeff());
    double jitter = kBaseJitter * diag_scale;

    for (int attempt = 0; attempt < kMaxJitterAttempts; ++attempt) {
        Eigen::MatrixXd candidate = sym_cov;
        candidate.diagonal().array() += jitter;

        Eigen::LLT<Eigen::MatrixXd> llt(candidate);
        if (llt.info() == Eigen::Success) {
            return llt.matrixL();
        }

        jitter *= 10.0;
    }

    throw std::runtime_error(
        "UnscentedTransform: covariance is not numerically positive definite"
    );
}

auto UnscentedTransform::generate_sigma_points(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const UnscentedTransformParameters& params
) -> std::vector<Eigen::VectorXd> {
    const int n = mean.size();
    if (covariance.rows() != n || covariance.cols() != n) {
        throw std::invalid_argument(
            "UnscentedTransform: covariance dimensions must match mean dimension"
        );
    }

    const double lambda = validate_parameters(n, params);
    const double scaling = static_cast<double>(n) + lambda;

    const Eigen::MatrixXd L = compute_cholesky_factor(scaling * covariance);

    std::vector<Eigen::VectorXd> sigma_points(static_cast<std::size_t>(2 * n + 1));
    sigma_points[0] = mean;

    for (int i = 0; i < n; ++i) {
        const Eigen::VectorXd offset = L.col(i);
        sigma_points[static_cast<std::size_t>(i + 1)] = mean + offset;
        sigma_points[static_cast<std::size_t>(i + 1 + n)] = mean - offset;
    }

    return sigma_points;
}

auto UnscentedTransform::transform_distribution(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const NonlinearTransformFunction& transform_fn,
    const Eigen::MatrixXd& additive_noise,
    const UnscentedTransformParameters& params
) -> TransformResult {
    if (!transform_fn) {
        throw std::invalid_argument("UnscentedTransform: transform function cannot be empty");
    }

    TransformResult result;
    result.weights = compute_weights(mean.size(), params);
    result.source_sigma_points = generate_sigma_points(mean, covariance, params);
    result.transformed_sigma_points.reserve(result.source_sigma_points.size());

    int transformed_dim = -1;
    for (const Eigen::VectorXd& sigma : result.source_sigma_points) {
        Eigen::VectorXd transformed = transform_fn(sigma);
        if (transformed_dim < 0) {
            transformed_dim = transformed.size();
        }
        if (transformed.size() != transformed_dim) {
            throw std::invalid_argument(
                "UnscentedTransform: transformed sigma points must have consistent dimension"
            );
        }
        result.transformed_sigma_points.push_back(std::move(transformed));
    }

    result.moments = recover_gaussian(
        result.transformed_sigma_points,
        result.weights,
        additive_noise
    );

    return result;
}

auto UnscentedTransform::recover_gaussian(
    const std::vector<Eigen::VectorXd>& sigma_points,
    const Weights& weights,
    const Eigen::MatrixXd& additive_noise
) -> Moments {
    if (sigma_points.empty()) {
        throw std::invalid_argument("UnscentedTransform: sigma points cannot be empty");
    }

    if (weights.mean.size() != static_cast<int>(sigma_points.size()) ||
        weights.covariance.size() != static_cast<int>(sigma_points.size())) {
        throw std::invalid_argument(
            "UnscentedTransform: weight count must match sigma point count"
        );
    }

    const int dim = sigma_points.front().size();
    for (const Eigen::VectorXd& sigma : sigma_points) {
        if (sigma.size() != dim) {
            throw std::invalid_argument(
                "UnscentedTransform: all sigma points must have same dimension"
            );
        }
    }

    if (additive_noise.size() != 0 &&
        (additive_noise.rows() != dim || additive_noise.cols() != dim)) {
        throw std::invalid_argument(
            "UnscentedTransform: additive noise dimensions must match transformed dimension"
        );
    }

    Moments moments;
    moments.mean = Eigen::VectorXd::Zero(dim);
    for (std::size_t i = 0; i < sigma_points.size(); ++i) {
        moments.mean += weights.mean(static_cast<int>(i)) * sigma_points[i];
    }

    moments.covariance = Eigen::MatrixXd::Zero(dim, dim);
    for (std::size_t i = 0; i < sigma_points.size(); ++i) {
        const Eigen::VectorXd diff = sigma_points[i] - moments.mean;
        moments.covariance +=
            weights.covariance(static_cast<int>(i)) * (diff * diff.transpose());
    }

    if (additive_noise.size() != 0) {
        moments.covariance += additive_noise;
    }

    moments.covariance = 0.5 * (moments.covariance + moments.covariance.transpose());
    return moments;
}

auto UnscentedTransform::compute_cross_covariance(
    const std::vector<Eigen::VectorXd>& source_sigma_points,
    const Eigen::VectorXd& source_mean,
    const std::vector<Eigen::VectorXd>& target_sigma_points,
    const Eigen::VectorXd& target_mean,
    const Weights& weights
) -> Eigen::MatrixXd {
    if (source_sigma_points.empty() || target_sigma_points.empty()) {
        throw std::invalid_argument("UnscentedTransform: sigma points cannot be empty");
    }

    if (source_sigma_points.size() != target_sigma_points.size() ||
        weights.covariance.size() != static_cast<int>(source_sigma_points.size())) {
        throw std::invalid_argument(
            "UnscentedTransform: source/target sigma point counts and weights must match"
        );
    }

    const int source_dim = source_mean.size();
    const int target_dim = target_mean.size();

    Eigen::MatrixXd cross_cov = Eigen::MatrixXd::Zero(source_dim, target_dim);

    for (std::size_t i = 0; i < source_sigma_points.size(); ++i) {
        if (source_sigma_points[i].size() != source_dim ||
            target_sigma_points[i].size() != target_dim) {
            throw std::invalid_argument(
                "UnscentedTransform: sigma point dimensions must match provided means"
            );
        }

        const Eigen::VectorXd dx = source_sigma_points[i] - source_mean;
        const Eigen::VectorXd dz = target_sigma_points[i] - target_mean;

        cross_cov += weights.covariance(static_cast<int>(i)) * (dx * dz.transpose());
    }

    return cross_cov;
}

} // namespace filtering
