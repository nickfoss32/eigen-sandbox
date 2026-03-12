#pragma once

#include <Eigen/Dense>

#include <functional>
#include <vector>

namespace filtering {

/// @brief Sigma-point construction method used by the Unscented Transform.
enum class UnscentedSigmaPointScheme {
    /// @brief Julier's spherical-simplex sigma set with n+2 sigma points.
    JulierSphericalSimplex,

    /// @brief Classic scaled symmetric sigma set with 2n+1 sigma points.
    ScaledSymmetric
};

/// @brief Tuning parameters for the Unscented Transform.
struct UnscentedTransformParameters {
    /// @brief Sigma-point construction scheme.
    UnscentedSigmaPointScheme sigma_point_scheme =
        UnscentedSigmaPointScheme::JulierSphericalSimplex;

    /// @brief Central sigma-point weight for Julier's spherical-simplex set.
    ///
    /// Must satisfy 0 <= simplex_weight_0 < 1. The remaining n+1 sigma points
    /// each receive weight (1 - simplex_weight_0) / (n + 1).
    double simplex_weight_0 = 0.5;

    /// @brief Primary spread parameter for the scaled symmetric sigma set.
    double alpha = 0.3;

    /// @brief Prior knowledge parameter for the scaled symmetric sigma set.
    double beta = 2.0;

    /// @brief Secondary spread parameter for the scaled symmetric sigma set.
    double kappa = 0.0;
};

/// @brief Standalone Unscented Transform utility.
///
/// Provides sigma-point generation and Gaussian moment recovery for nonlinear
/// transforms. It can be used directly or by filters (e.g., UKF).
class UnscentedTransform {
public:
    /// @brief Callback signature for nonlinear sigma-point mapping.
    using NonlinearTransformFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

    /// @brief UT weight vectors for mean and covariance accumulation.
    struct Weights {
        /// @brief Mean weights for each sigma point.
        Eigen::VectorXd mean;

        /// @brief Covariance weights for each sigma point.
        Eigen::VectorXd covariance;

        /// @brief Lambda for the scaled symmetric scheme (0 for spherical simplex).
        double lambda = 0.0;
    };

    /// @brief Recovered Gaussian moments after transformation.
    struct Moments {
        /// @brief Recovered mean vector.
        Eigen::VectorXd mean;

        /// @brief Recovered covariance matrix.
        Eigen::MatrixXd covariance;
    };

    /// @brief Result container for a one-call UT round.
    struct TransformResult {
        /// @brief Weights used during this UT round.
        Weights weights;

        /// @brief Sigma points generated from input mean/covariance.
        std::vector<Eigen::VectorXd> source_sigma_points;

        /// @brief Sigma points after applying the nonlinear transform.
        std::vector<Eigen::VectorXd> transformed_sigma_points;

        /// @brief Recovered transformed-space mean/covariance.
        Moments moments;
    };

    /// @brief Compute UT weights for a given state dimension.
    /// @param state_dim Dimension of the state space.
    /// @param params Unscented transform tuning parameters.
    /// @return Weight vectors and lambda scalar.
    static auto compute_weights(
        int state_dim,
        const UnscentedTransformParameters& params = {}
    ) -> Weights;

    /// @brief Generate sigma points around a Gaussian (mean, covariance).
    /// @param mean Mean vector of the Gaussian.
    /// @param covariance Covariance matrix of the Gaussian.
    /// @param params Unscented transform tuning parameters.
    /// @return Sigma points with scheme-dependent size (n+2 or 2n+1).
    static auto generate_sigma_points(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance,
        const UnscentedTransformParameters& params = {}
    ) -> std::vector<Eigen::VectorXd>;

    /// @brief Execute a complete UT round in one call.
    ///
    /// This method performs:
    /// 1. Weight computation
    /// 2. Sigma-point generation
    /// 3. Nonlinear transformation of sigma points
    /// 4. Moment recovery (optionally with additive transformed-space noise)
    ///
    /// @param mean Input Gaussian mean.
    /// @param covariance Input Gaussian covariance.
    /// @param transform_fn Nonlinear mapping y = f(x) applied to each sigma point.
    /// @param additive_noise Optional additive covariance in transformed space.
    /// @param params Unscented transform tuning parameters.
    /// @return TransformResult containing sigma points, weights, and recovered moments.
    static auto transform_distribution(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance,
        const NonlinearTransformFunction& transform_fn,
        const Eigen::MatrixXd& additive_noise = Eigen::MatrixXd(),
        const UnscentedTransformParameters& params = {}
    ) -> TransformResult;

    /// @brief Recover Gaussian moments from transformed sigma points.
    /// @param sigma_points Transformed sigma points.
    /// @param weights UT weights matching sigma point count.
    /// @param additive_noise Optional additive covariance in transformed space.
    /// @return Recovered mean and covariance.
    static auto recover_gaussian(
        const std::vector<Eigen::VectorXd>& sigma_points,
        const Weights& weights,
        const Eigen::MatrixXd& additive_noise = Eigen::MatrixXd()
    ) -> Moments;

    /// @brief Compute cross-covariance between source and transformed spaces.
    /// @param source_sigma_points Sigma points in source space.
    /// @param source_mean Mean in source space.
    /// @param target_sigma_points Corresponding sigma points in target space.
    /// @param target_mean Mean in target space.
    /// @param weights Covariance weights for accumulation.
    /// @return Cross-covariance matrix E[(x-xbar)(y-ybar)^T].
    static auto compute_cross_covariance(
        const std::vector<Eigen::VectorXd>& source_sigma_points,
        const Eigen::VectorXd& source_mean,
        const std::vector<Eigen::VectorXd>& target_sigma_points,
        const Eigen::VectorXd& target_mean,
        const Weights& weights
    ) -> Eigen::MatrixXd;

private:
    /// @brief Validate UT parameters for a given dimension.
    /// @param state_dim Dimension of the state space.
    /// @param params Unscented transform tuning parameters.
    /// @return Lambda value.
    static auto validate_parameters(
        int state_dim,
        const UnscentedTransformParameters& params
    ) -> double;

    /// @brief Generate unit sigma points for the selected sigma-point scheme.
    /// @param state_dim Dimension of the state space.
    /// @param params Unscented transform tuning parameters.
    /// @return Unit sigma points whose weighted covariance is identity.
    static auto generate_unit_sigma_points(
        int state_dim,
        const UnscentedTransformParameters& params
    ) -> std::vector<Eigen::VectorXd>;

    /// @brief Compute a numerically robust Cholesky factor.
    /// @param covariance Symmetric covariance matrix.
    /// @return Lower-triangular Cholesky factor.
    static auto compute_cholesky_factor(const Eigen::MatrixXd& covariance) -> Eigen::MatrixXd;
};

} // namespace filtering
