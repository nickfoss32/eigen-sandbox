#include "dynamics/forces/time_window_force.hpp"

#include <stdexcept>

namespace dynamics {

TimeWindowForce::TimeWindowForce(
    std::shared_ptr<IForce> inner_force,
    double start_time,
    double end_time
) : inner_force_(std::move(inner_force)),
    start_time_(start_time),
    end_time_(end_time) {
    if (!inner_force_) {
        throw std::invalid_argument("TimeWindowForce: inner force cannot be null");
    }
    if (!(end_time_ > start_time_)) {
        throw std::invalid_argument("TimeWindowForce: end time must be greater than start time");
    }
}

auto TimeWindowForce::compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d {
    if (!is_active(ctx.t)) {
        return Eigen::Vector3d::Zero();
    }
    return inner_force_->compute_acceleration(ctx);
}

auto TimeWindowForce::compute_jacobian(const ForceContext& ctx) const
    -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {
    if (!is_active(ctx.t)) {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    return inner_force_->compute_jacobian(ctx);
}

auto TimeWindowForce::is_active(double time_seconds) const -> bool {
    return time_seconds >= start_time_ && time_seconds < end_time_;
}

} // namespace dynamics
