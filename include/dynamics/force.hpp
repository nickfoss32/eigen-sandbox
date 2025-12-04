#pragma once

#include <Eigen/Dense>

namespace dynamics {

/// @brief Context information available to all force models
struct ForceContext {
    double t;                   ///< Time (s)
    Eigen::Vector3d position;   ///< Position vector r (m)
    Eigen::Vector3d velocity;   ///< Velocity vector v (m/s)
};

/// @brief Interface for forces affecting translational motion
/// 
/// @details This interface represents any force that produces an acceleration
///          as a function of state and time:
///          
///          **a** = f(**r**, **v**, t)
///          
///          where:
///          - **a** ∈ ℝ³ is the acceleration vector (m/s²)
///          - **r** ∈ ℝ³ is the position vector (m)
///          - **v** ∈ ℝ³ is the velocity vector (m/s)
///          - t ∈ ℝ is time (s)
///          
///          Most forces are **autonomous** (time-independent), such as:
///          - Gravity: f(r, v) only
///          - Drag: f(r, v) only
///          - Spring forces: f(r, v) only
///          
///          However, time is provided for forces that explicitly depend on it:
///          - Thrust schedules: f(r, v, t) with time-varying throttle
///          - Solar pressure: f(r, v, t) with time-varying sun position
///          - Scheduled maneuvers: f(r, v, t) with burn windows
///          
///          Implementations must provide both the force acceleration f(r,v,t)
///          and its Jacobian matrices ∂f/∂r and ∂f/∂v for use in the
///          Extended Kalman Filter.
///          
/// @note Time-independent forces may ignore ctx.t in their implementation.
/// @note Some forces may not depend on all state variables (e.g., gravity is
///       velocity-independent). Unused Jacobians should return zero matrices.
class IForce {
public:
    /// @brief Virtual destructor
    virtual ~IForce() = default;
    
    /// @brief Computes the force acceleration a = f(r, v, t)
    /// 
    /// @details Returns the instantaneous acceleration due to this force
    ///          evaluated at the given state (position, velocity) and time.
    ///          
    ///          **a** = f(**r**, **v**, t)
    ///          
    ///          For time-independent (autonomous) forces, t may be ignored.
    ///          
    /// @param ctx Force context containing time, position, and velocity
    /// @return Acceleration vector (m/s²)
    virtual Eigen::Vector3d compute_force(const ForceContext& ctx) const = 0;
    
    /// @brief Computes the Jacobian matrices ∂f/∂r and ∂f/∂v
    /// 
    /// @details Returns the partial derivatives of the force acceleration
    ///          with respect to position and velocity:
    ///          
    ///          ∂**a**/∂**r** = ∂f/∂**r** ∈ ℝ³ˣ³
    ///          ∂**a**/∂**v** = ∂f/∂**v** ∈ ℝ³ˣ³
    ///          
    ///          These Jacobians are used by the Extended Kalman Filter to
    ///          linearize the dynamics and propagate covariance.
    ///          
    /// @param ctx Force context containing time, position, and velocity
    /// @return Pair of Jacobian matrices (∂f/∂r, ∂f/∂v)
    /// @note If force is position-independent, return ∂f/∂r = 0
    /// @note If force is velocity-independent, return ∂f/∂v = 0
    /// @note Time derivatives ∂f/∂t are not needed (used only for explicit time dependence)
    virtual std::pair<Eigen::Matrix3d, Eigen::Matrix3d> compute_jacobian(
        const ForceContext& ctx
    ) const = 0;
};

} // namespace dynamics
