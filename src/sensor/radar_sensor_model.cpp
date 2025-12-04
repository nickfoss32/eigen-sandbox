#include "sensor/radar_sensor_model.hpp"

#include <cmath>
#include <random>

namespace sensor {

RadarSensorModel::RadarSensorModel(
    const Eigen::Vector3d& radar_position,
    double range_noise,
    double angle_noise
) : radar_position_(radar_position)
{
    R_ = Eigen::Matrix3d::Zero();
    R_(0, 0) = range_noise * range_noise;
    R_(1, 1) = angle_noise * angle_noise;
    R_(2, 2) = angle_noise * angle_noise;
}

Eigen::VectorXd RadarSensorModel::compute_measurement(const SensorContext& ctx) const {
    // Extract target position (first 3 elements of state)
    Eigen::Vector3d target_pos = ctx.state.head<3>();
    
    // Relative position: target - radar
    Eigen::Vector3d rel_pos = target_pos - radar_position_;
    
    // Spherical coordinates
    double range = rel_pos.norm();
    double azimuth = std::atan2(rel_pos(1), rel_pos(0));  // Angle in xy-plane
    double elevation = std::asin(rel_pos(2) / range);      // Angle from xy-plane
    
    Eigen::Vector3d measurement;
    measurement << range, azimuth, elevation;
    return measurement;
}

Eigen::MatrixXd RadarSensorModel::compute_jacobian(const SensorContext& ctx) const {
    // Extract target position (first 3 elements of state)
    Eigen::Vector3d target_pos = ctx.state.head<3>();
    
    // Relative position: Δr = target - radar
    Eigen::Vector3d dr = target_pos - radar_position_;
    double dx = dr(0);
    double dy = dr(1);
    double dz = dr(2);
    
    // Precompute commonly used terms
    double range = dr.norm();
    double range_sq = range * range;
    double range_xy_sq = dx * dx + dy * dy;
    double range_xy = std::sqrt(range_xy_sq);
    
    // Avoid division by zero
    const double eps = 1e-12;
    if (range < eps || range_xy < eps) {
        // Return zero Jacobian if at singular point
        int state_dim = ctx.state.size();
        return Eigen::MatrixXd::Zero(3, state_dim);
    }
    
    // ========================================
    // Jacobian H = ∂h/∂x where h = [range, azimuth, elevation]'
    // ========================================
    
    // Measurement: z = [ρ, α, ε]'
    // where:
    //   ρ = range = sqrt(dx² + dy² + dz²)
    //   α = azimuth = atan2(dy, dx)
    //   ε = elevation = asin(dz / ρ)
    
    // ∂ρ/∂x = dx / ρ
    // ∂ρ/∂y = dy / ρ
    // ∂ρ/∂z = dz / ρ
    double drange_dx = dx / range;
    double drange_dy = dy / range;
    double drange_dz = dz / range;
    
    // ∂α/∂x = -dy / (dx² + dy²)
    // ∂α/∂y =  dx / (dx² + dy²)
    // ∂α/∂z =  0
    double daz_dx = -dy / range_xy_sq;
    double daz_dy =  dx / range_xy_sq;
    double daz_dz =  0.0;
    
    // ∂ε/∂x = -dx*dz / (ρ² * sqrt(dx² + dy²))
    // ∂ε/∂y = -dy*dz / (ρ² * sqrt(dx² + dy²))
    // ∂ε/∂z = sqrt(dx² + dy²) / ρ²
    double del_dx = -dx * dz / (range_sq * range_xy);
    double del_dy = -dy * dz / (range_sq * range_xy);
    double del_dz =  range_xy / range_sq;
    
    // Build Jacobian matrix H
    // H is 3 x state_dim, but only first 3 columns are non-zero
    // (measurement only depends on position, not velocity)
    int state_dim = ctx.state.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, state_dim);
    
    // ∂h/∂position (first 3 columns)
    H(0, 0) = drange_dx;  // ∂ρ/∂x
    H(0, 1) = drange_dy;  // ∂ρ/∂y
    H(0, 2) = drange_dz;  // ∂ρ/∂z
    
    H(1, 0) = daz_dx;     // ∂α/∂x
    H(1, 1) = daz_dy;     // ∂α/∂y
    H(1, 2) = daz_dz;     // ∂α/∂z
    
    H(2, 0) = del_dx;     // ∂ε/∂x
    H(2, 1) = del_dy;     // ∂ε/∂y
    H(2, 2) = del_dz;     // ∂ε/∂z
    
    // ∂h/∂velocity = 0 (columns 4-6 remain zero)
    
    return H;
}

Eigen::MatrixXd RadarSensorModel::get_noise_covariance() const {
    return R_;
}

} // namespace sensor
