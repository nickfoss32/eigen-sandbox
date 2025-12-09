#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <vector>
#include <stdexcept>

namespace fitting {

/// @brief Common base class for all plane fitters
class PlaneFitter {
public:
    /// @brief Constructor
    /// @param fitThroughOrigin Whether or not the calculated plane fit should go through the origin
    PlaneFitter(bool fitThroughOrigin = true);

    /// @brief Virtual destructor for proper cleanup
    virtual ~PlaneFitter() = default;

    /// @brief Interface for calculating the best-fit plane for a given set of points.
    /// @note Concrete, non-virtual interface for computeFit.
    /// @param points The set of positions to calculate a best-fit plane for.
    /// @return the plane normal and a point on the plane.
    auto computeFit(const std::vector<Eigen::Vector3d>& points) const -> std::pair<Eigen::Vector3d, Eigen::Vector3d>;
    
    /// @brief Projects a state vector onto the plane defined by the normal and a point on the plane.
    /// Assumes the state is a 6D vector: [position (3D), velocity (3D)].
    /// Projects the position to the closest point on the plane and the velocity to the plane's tangent space.
    /// @param state The input state vector
    /// @param plane_normal The vector representing the plane's normal
    /// @param point_on_plane A point on the plane
    /// @return The state projected onto the plane
    auto projectState(const Eigen::VectorXd& state, const Eigen::Vector3d& plane_normal, const Eigen::Vector3d& point_on_plane) const -> Eigen::VectorXd;

    /// @brief Boolean indicating whether fit should go through the origin (0,0,0)
    const bool fitThroughOrigin_{true};

private:
    /// @brief Virtual interface for computing a best fit plane
    /// @param points The set of positions to calculate a best-fit plane for.
    /// @return the plane normal and a point on the plane.
    virtual std::pair<Eigen::Vector3d, Eigen::Vector3d> doComputeFit(const std::vector<Eigen::Vector3d>& points) const = 0;
};

/// @brief Plane fitter using ordinary least squares (OLS) regression, assuming z as the dependent variable.
/// Fits a plane of the form z = a*x + b*y + c, which minimizes squared errors in the z-direction.
/// @note This may not perform well for planes nearly parallel to the z-axis.
class RegressionPlaneFitter : public PlaneFitter
{
public:
    /// @brief Constructor
    /// @param fitThroughOrigin Whether or not the calculated plane fit should go through the origin
    RegressionPlaneFitter(bool fitThroughOrigin = true);

private:
    /// @brief Calculates the best-fit plane for a given set of points.
    /// @return the plane normal and a point on the plane.
    /// @throws std::runtime_error if plane is nearly paralell to the z-axis.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> doComputeFit(const std::vector<Eigen::Vector3d>& points) const override;
};

/// @brief Plane fitter using total least squares (TLS, orthogonal regression), minimizing perpendicular distances.
/// Uses SVD on centered points to find the plane normal as the direction of least variance.
class TotalLeastSquaresPlaneFitter : public PlaneFitter {
public:
    /// @brief Constructor
    /// @param fitThroughOrigin Whether or not the calculated plane fit should go through the origin
    TotalLeastSquaresPlaneFitter(bool fitThroughOrigin = true);

private:
    /// @brief Calculates the best-fit plane for a given set of points.
    /// @return the plane normal and a point on the plane.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> doComputeFit(const std::vector<Eigen::Vector3d>& points) const override;
};
} // namespace fitting
