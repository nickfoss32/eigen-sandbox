#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <stdexcept>

/// @brief Common base class for all plane fitters
class PlaneFitter {
public:
    /// @brief Constructor
    /// @param fitThroughOrigin Whether or not the calculated plane fit should go through the origin
    PlaneFitter(bool fitThroughOrigin = true)
     : fitThroughOrigin_(fitThroughOrigin)
    {}

    /// @brief Virtual destructor for proper cleanup
    virtual ~PlaneFitter() = default;

    /// @brief Interface for calculating the best-fit plane for a given set of points.
    /// @note Concrete, non-virtual interface for computeFit.
    /// @param points The set of positions to calculate a best-fit plane for.
    /// @return the plane normal and a point on the plane.
    auto computeFit(const std::vector<Eigen::Vector3d>& points) const -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
        size_t n = points.size();
        if (n < 3) {
            throw std::invalid_argument("At least 3 points are required for plane fitting.");
        }
        
        return doComputeFit(points);
    };
    
    /// @brief Projects a state vector onto the plane defined by the normal and a point on the plane.
    /// Assumes the state is a 6D vector: [position (3D), velocity (3D)].
    /// Projects the position to the closest point on the plane and the velocity to the plane's tangent space.
    /// @param state The input state vector
    /// @param plane_normal The vector representing the plane's normal
    /// @param point_on_plane A point on the plane
    /// @return The state projected onto the plane
    auto projectState(const Eigen::VectorXd& state, const Eigen::Vector3d& plane_normal, const Eigen::Vector3d& point_on_plane) const
    {
        if (state.size() != 6) {
            throw std::invalid_argument("State vector must be 6-dimensional (position + velocity).");
        }

        Eigen::Vector3d pos = state.head<3>();
        Eigen::Vector3d vel = state.tail<3>();

        Eigen::Vector3d n = plane_normal.normalized();  // Ensure unit normal

        double dist = n.dot(pos - point_on_plane);
        Eigen::Vector3d pos_proj = pos - dist * n;

        Eigen::Vector3d vel_proj = vel - n.dot(vel) * n;

        Eigen::VectorXd state_proj(6);
        state_proj.head<3>() = pos_proj;
        state_proj.tail<3>() = vel_proj;

        return state_proj;
    }

protected:
    /// @brief Boolean indicating whether fit should go through the origin (0,0,0)
    bool fitThroughOrigin_{true};

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
    RegressionPlaneFitter(bool fitThroughOrigin = true)
     : PlaneFitter(fitThroughOrigin)
    {}

private:
    /// @brief Calculates the best-fit plane for a given set of points.
    /// @return the plane normal and a point on the plane.
    /// @throws std::runtime_error if plane is nearly paralell to the z-axis.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> doComputeFit(const std::vector<Eigen::Vector3d>& points) const override
    {
        size_t n = points.size();

        Eigen::MatrixXd A(n, 3);
        Eigen::VectorXd z(n);
        Eigen::Vector3d avg = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < n; ++i) {
            const auto& p = points[i];
            A(i, 0) = p.x();
            A(i, 1) = p.y();
            A(i, 2) = 1.0;
            z(i) = p.z();
            avg += p;
        }
        avg /= static_cast<double>(n);

        // Solve for theta = [a, b, c]^T using QR decomposition
        Eigen::Vector3d theta = A.householderQr().solve(z);
        double a = theta[0];
        double b = theta[1];
        double c = theta[2];
        if (!PlaneFitter::fitThroughOrigin_) {
            c = 0.0;
        }

        // Form normal vector for plane a*x + b*y - z + c = 0
        Eigen::Vector3d normal(a, b, -1.0);
        double normal_norm = normal.norm();
        if (normal_norm < 1e-6) {
            throw std::runtime_error("Computed normal is near-zero, indicating numerical instability (plane may be nearly parallel to z-axis).");
        }
        normal /= normal_norm; // Normalize

        // Compute a point on the plane: use average x,y with fitted z
        double z_fit = a * avg.x() + b * avg.y() + c;
        Eigen::Vector3d point_on_plane(avg.x(), avg.y(), z_fit);
        if (PlaneFitter::fitThroughOrigin_) {
            point_on_plane = Eigen::Vector3d::Zero();
        }

        return {normal, point_on_plane};
    }
};

/// @brief Plane fitter using total least squares (TLS, orthogonal regression), minimizing perpendicular distances.
/// Uses SVD on centered points to find the plane normal as the direction of least variance.
class TotalLeastSquaresPlaneFitter : public PlaneFitter {
public:
    /// @brief Constructor
    /// @param fitThroughOrigin Whether or not the calculated plane fit should go through the origin
    TotalLeastSquaresPlaneFitter(bool fitThroughOrigin = true)
    : PlaneFitter(fitThroughOrigin)
    {}

private:
    /// @brief Calculates the best-fit plane for a given set of points.
    /// @return the plane normal and a point on the plane.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> doComputeFit(const std::vector<Eigen::Vector3d>& points) const override
    {
        size_t n = points.size();

        // Compute centroid
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& p : points) {
            centroid += p;
        }
        centroid /= static_cast<double>(n);

        // Form centered data matrix (n x 3)
        Eigen::MatrixXd centered(n, 3);
        for (size_t i = 0; i < n; ++i) {
            if (PlaneFitter::fitThroughOrigin_) {
                centered.row(i) = points[i];
            } else {
                centered.row(i) = points[i] - centroid;
            }
        }

        // Compute SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Normal is the right singular vector corresponding to the smallest singular value (last column)
        Eigen::Vector3d normal = svd.matrixV().col(2).normalized();

        if (PlaneFitter::fitThroughOrigin_) {
            return {normal, Eigen::Vector3d::Zero()};
        } else {
            return {normal, centroid};
        }
    }
};
