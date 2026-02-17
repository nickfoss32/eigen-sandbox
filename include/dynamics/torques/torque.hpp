#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace dynamics {

/// @brief Context information available to all torque models
struct TorqueContext {
    double t;                          ///< Time (s)
    Eigen::Vector3d position;          ///< Position vector r in inertial frame (m)
    Eigen::Vector3d velocity;          ///< Velocity vector v in inertial frame (m/s)
    Eigen::Quaterniond orientation;    ///< Orientation quaternion q (inertial to body)
    Eigen::Vector3d angular_velocity;  ///< Angular velocity ω in body frame (rad/s)
};

/// @brief Interface for torques affecting rotational motion
/// 
/// @details This interface represents any torque that produces an angular
///          acceleration as a function of state and time:
///          
///          **τ** = f(**q**, **ω**, **r**, **v**, t)
///          
///          where:
///          - **τ** ∈ ℝ³ is the torque vector in body frame (N·m)
///          - **q** ∈ S³ is the orientation quaternion (inertial to body)
///          - **ω** ∈ ℝ³ is the angular velocity in body frame (rad/s)
///          - **r** ∈ ℝ³ is the position vector in inertial frame (m)
///          - **v** ∈ ℝ³ is the velocity vector in inertial frame (m/s)
///          - t ∈ ℝ is time (s)
///          
///          This interface is only applicable to **rigid bodies** with rotational
///          degrees of freedom. Point masses have no rotational dynamics.
///          
///          Common torque dependencies:
///          - **Gravity gradient**: f(q, r) - depends on orientation and orbital position
///          - **Aerodynamic moments**: f(q, ω, v) - depends on attitude and velocity
///          - **Control torques**: f(q, ω, t) - reaction wheels, thrusters, magnetorquers
///          - **Solar pressure moments**: f(q, r, t) - depends on sun position
///          
///          Implementations must provide the torque function f(q,ω,r,v,t) and
///          its Jacobian matrices for use in the Extended Kalman Filter for
///          attitude estimation.
///          
/// @note Torques are returned in the **body frame** (aligned with principal axes).
///       This is the natural frame for Euler's equation: ω̇ = I⁻¹·(τ - ω×I·ω)
/// @note Time-independent torques may ignore ctx.t in their implementation.
/// @note Torques that don't depend on orbital state may ignore ctx.position and ctx.velocity.
/// @see IForce for translational dynamics (forces on center of mass)
class ITorque {
public:
    /// @brief Virtual destructor
    virtual ~ITorque() = default;

    /// @brief Computes the torque τ = f(q, ω, r, v, t)
    /// 
    /// @details Returns the instantaneous torque acting on the rigid body
    ///          evaluated at the given attitude (orientation, angular velocity),
    ///          orbital state (position, velocity), and time.
    ///          
    ///          **τ** = f(**q**, **ω**, **r**, **v**, t)
    ///          
    ///          The torque is expressed in the **body frame** and is used in
    ///          Euler's rotational equation of motion:
    ///          
    ///          **ω̇** = I⁻¹·(**τ** - **ω** × I·**ω**)
    ///          
    /// @param ctx Torque context containing time, orbital state, and attitude state
    /// @return Torque vector in body frame (N·m)
    virtual auto compute_torque(const TorqueContext& ctx) const -> Eigen::Vector3d = 0;
    
    /// @brief Computes the Jacobian matrices ∂τ/∂q, ∂τ/∂ω, ∂τ/∂r, ∂τ/∂v
    /// 
    /// @details Returns the partial derivatives of the torque with respect to
    ///          attitude state (orientation, angular velocity) and orbital state
    ///          (position, velocity):
    ///          
    ///          ∂**τ**/∂**q_vec** ∈ ℝ³ˣ³  (using 3-parameter rotation vector)
    ///          ∂**τ**/∂**ω** ∈ ℝ³ˣ³
    ///          ∂**τ**/∂**r** ∈ ℝ³ˣ³
    ///          ∂**τ**/∂**v** ∈ ℝ³ˣ³
    ///          
    ///          These Jacobians are used by the Extended Kalman Filter to
    ///          linearize the attitude dynamics and propagate covariance.
    ///          
    ///          The quaternion Jacobian is returned with respect to a 3-parameter
    ///          error representation (e.g., Gibbs vector or MRP) rather than the
    ///          4-parameter quaternion directly, to avoid singularity issues.
    ///          
    /// @param ctx Torque context containing time, orbital state, and attitude state
    /// @return Tuple of Jacobian matrices (∂τ/∂q_vec, ∂τ/∂ω, ∂τ/∂r, ∂τ/∂v)
    /// @note If torque is orientation-independent, return ∂τ/∂q = 0
    /// @note If torque is angular-velocity-independent, return ∂τ/∂ω = 0
    /// @note If torque is position-independent, return ∂τ/∂r = 0
    /// @note If torque is velocity-independent, return ∂τ/∂v = 0
    virtual auto compute_jacobian(const TorqueContext& ctx) const -> std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d> = 0;
};

} // namespace dynamics
