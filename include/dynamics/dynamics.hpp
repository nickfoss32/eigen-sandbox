#pragma once

#include <Eigen/Dense>
#include <optional>

namespace dynamics {

/// @brief Unified dynamics interface
/// 
/// @details All dynamics systems must implement the state-space form:
///          
///          ẋ = f(t, x)
///          
///          where:
///          - x ∈ ℝⁿ is the state vector
///          - f: ℝ × ℝⁿ → ℝⁿ is the dynamics function
///          - ẋ = dx/dt is the state derivative
///          
///          The dynamics function f(t, x) and its Jacobian ∂f/∂x are required
///          for numerical integration and Extended Kalman Filtering.
///          
///          Some dynamics may optionally provide analytical solutions for
///          improved performance in special cases.
class IDynamics {
public:
    virtual ~IDynamics() = default;
    
    /// @brief Compute the dynamics function: ẋ = f(t, x)
    /// 
    /// @details Returns the state derivative (rate of change) at the given
    ///          time and state. This is the core function that defines the
    ///          dynamics of the system.
    ///          
    ///          Used by:
    ///          - Numerical integrators (RK4, RK45, etc.) for propagation
    ///          - EKF for nonlinear dynamics evaluation
    ///          
    /// @param t Time (s)
    /// @param state Current state vector x ∈ ℝⁿ
    /// @return State derivative ẋ = f(t, x) ∈ ℝⁿ
    virtual auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd = 0;
    
    /// @brief Compute the Jacobian of the dynamics: F = ∂f/∂x
    /// 
    /// @details Returns the linearization of the dynamics about the given state.
    ///          This is used by the Extended Kalman Filter to propagate covariance
    ///          and by some advanced integrators for improved accuracy.
    ///          
    ///          For state x ∈ ℝⁿ, returns n×n matrix F where:
    ///          F_ij = ∂f_i/∂x_j
    ///          
    /// @param t Time (s)
    /// @param state Current state vector x ∈ ℝⁿ
    /// @return Jacobian matrix F = ∂f/∂x ∈ ℝⁿˣⁿ
    virtual auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd = 0;
    
    /// @brief Get dimension of state vector
    /// @return State dimension n
    virtual auto get_state_dimension() const -> int = 0;

    // ========================================
    // OPTIONAL: Analytical Solution Support
    // ========================================
    
    /// @brief Check if analytical solution is available
    /// @return true if solve_analytical() and compute_analytical_stm() are implemented
    virtual auto has_analytical_solution() const -> bool {
        return false;  // Default: no analytical solution
    }
    
    /// @brief Solve dynamics analytically from t0 to tf
    /// 
    /// @details For some special cases (e.g., Keplerian orbits, linear systems),
    ///          the dynamics can be solved analytically without numerical integration.
    ///          This is much faster and more accurate when available.
    ///          
    /// @param t0 Initial time
    /// @param state0 Initial state x(t0)
    /// @param tf Final time
    /// @return Final state x(tf) if analytical solution available, std::nullopt otherwise
    virtual auto solve_analytical(double t0, const Eigen::VectorXd& state0, double tf) const -> std::optional<Eigen::VectorXd> {
        return std::nullopt;
    }
    
    /// @brief Compute state transition matrix analytically: Φ(tf, t0) = ∂x(tf)/∂x(t0)
    /// 
    /// @details The STM describes how perturbations in initial state affect the final state.
    ///          For linear systems: x(tf) = Φ(tf, t0)·x(t0)
    ///          For nonlinear systems: δx(tf) ≈ Φ(tf, t0)·δx(t0)
    ///          
    ///          Used by the EKF for covariance propagation.
    ///          
    /// @param t0 Initial time
    /// @param state0 Initial state x(t0)
    /// @param tf Final time
    /// @return STM Φ(tf, t0) if analytical solution available, std::nullopt otherwise
    virtual auto compute_analytical_stm(double t0, const Eigen::VectorXd& state0, double tf) const -> std::optional<Eigen::MatrixXd> {
        return std::nullopt;
    }
};

} // namespace dynamics
