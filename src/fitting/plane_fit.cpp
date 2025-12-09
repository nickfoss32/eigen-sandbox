#include "fitting/plane_fit.hpp"

namespace fitting {
PlaneFitter::PlaneFitter(bool fitThroughOrigin)
: fitThroughOrigin_(fitThroughOrigin)
{}

auto PlaneFitter::computeFit(const std::vector<Eigen::Vector3d>& points) const -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
    size_t n = points.size();
    if (n < 3) {
        throw std::invalid_argument("At least 3 points are required for plane fitting.");
    }
    
    return doComputeFit(points);
};

auto PlaneFitter::projectState(const Eigen::VectorXd& state, const Eigen::Vector3d& plane_normal, const Eigen::Vector3d& point_on_plane) const -> Eigen::VectorXd
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
    vel_proj = vel_proj.normalized() * vel.norm();

    Eigen::VectorXd state_proj(6);
    state_proj.head<3>() = pos_proj;
    state_proj.tail<3>() = vel_proj;

    return state_proj;
}

RegressionPlaneFitter::RegressionPlaneFitter(bool fitThroughOrigin)
: PlaneFitter(fitThroughOrigin)
{}

std::pair<Eigen::Vector3d, Eigen::Vector3d> RegressionPlaneFitter::doComputeFit(const std::vector<Eigen::Vector3d>& points) const
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

TotalLeastSquaresPlaneFitter::TotalLeastSquaresPlaneFitter(bool fitThroughOrigin)
: PlaneFitter(fitThroughOrigin)
{}

std::pair<Eigen::Vector3d, Eigen::Vector3d> TotalLeastSquaresPlaneFitter::doComputeFit(const std::vector<Eigen::Vector3d>& points) const
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
} // namespace fitting
