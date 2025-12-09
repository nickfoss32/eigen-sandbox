#include <gtest/gtest.h>

#include "filtering/extended_kalman_filter.hpp"
#include "propagator/numerical_propagator.hpp"
#include "dynamics/point_mass_dynamics.hpp"
#include "dynamics/gravity.hpp"
#include "integrator/rk4.hpp"
#include "sensor/radar_sensor_model.hpp"
#include "common/types.hpp"

#include <Eigen/Dense>

#include <memory>
#include <cmath>
#include <random>

class ExtendedKalmanFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initial state: Low Earth Orbit at 400km altitude
        double earth_radius = 6.371e6;  // meters
        double altitude = 400e3;         // 400 km
        double orbital_radius = earth_radius + altitude;  // ~6.771e6 m
        
        x0_ = Eigen::VectorXd(6);
        x0_ << orbital_radius, 0.0, 0.0,  // Position: on x-axis at orbital radius
            0.0, 7670.0, 0.0;          // Velocity: ~7.67 km/s orbital velocity
        
        // Initial covariance
        P0_ = Eigen::MatrixXd::Identity(6, 6);
        P0_.block<3,3>(0,0) *= 100.0 * 100.0;   // ±100m position uncertainty
        P0_.block<3,3>(3,3) *= 10.0 * 10.0;     // ±10 m/s velocity uncertainty
        
        // Setup dynamics (point mass with gravity)
        auto gravity = std::make_shared<dynamics::PointMassGravity>();
        dynamics_ = std::make_shared<dynamics::PointMassDynamics>(
            std::vector<std::shared_ptr<dynamics::IForce>>{gravity}
        );
        
        // Setup propagator
        auto integrator = std::make_shared<integrator::RK4Integrator>();
        propagator_ = std::make_shared<propagator::NumericalPropagator>(
            dynamics_, integrator, 0.1  // 0.1 second integration steps
        );
        
        // Setup radar sensor at origin
        Eigen::Vector3d radar_pos(0.0, 0.0, 0.0);
        sensor_model_ = std::make_shared<sensor::RadarSensorModel>(
            radar_pos, 10.0, 0.001  // 10m range noise, 0.001 rad angle noise
        );
        
        // Process noise
        Q_func_ = [](double dt) {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
            Q.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * 1.0 * dt;
            Q.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * 0.1 * dt;
            return Q;
        };
    }
    
    Eigen::VectorXd x0_;
    Eigen::MatrixXd P0_;
    std::shared_ptr<dynamics::PointMassDynamics> dynamics_;
    std::shared_ptr<propagator::NumericalPropagator> propagator_;
    std::shared_ptr<sensor::RadarSensorModel> sensor_model_;
    filtering::ExtendedKalmanFilter::ProcessNoiseFunction Q_func_;
};

// ========================================
// Constructor Tests
// ========================================

TEST_F(ExtendedKalmanFilterTest, ConstructorInitializesState) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    Eigen::VectorXd state = ekf.get_state();
    EXPECT_EQ(state.size(), 6);

    double earth_radius = 6.371e6;
    double altitude = 400e3;
    double orbital_radius = earth_radius + altitude;
    
    EXPECT_NEAR(state(0), orbital_radius, 1.0);  // x position
    EXPECT_DOUBLE_EQ(state(1), 0.0);             // y position
    EXPECT_DOUBLE_EQ(state(2), 0.0);             // z position
}

TEST_F(ExtendedKalmanFilterTest, ConstructorInitializesCovariance) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    Eigen::MatrixXd P = ekf.get_covariance();
    EXPECT_EQ(P.rows(), 6);
    EXPECT_EQ(P.cols(), 6);
    EXPECT_DOUBLE_EQ(P(0,0), 100.0 * 100.0);
    EXPECT_DOUBLE_EQ(P(3,3), 10.0 * 10.0);
}

TEST_F(ExtendedKalmanFilterTest, ConstructorThrowsOnDimensionMismatch) {
    Eigen::MatrixXd bad_P = Eigen::MatrixXd::Identity(5, 5);  // Wrong size
    
    EXPECT_THROW(
        filtering::ExtendedKalmanFilter ekf(
            x0_, bad_P, propagator_, sensor_model_, Q_func_, 0.0
        ),
        std::invalid_argument
    );
}

TEST_F(ExtendedKalmanFilterTest, ConstructorThrowsOnNullPropagator) {
    EXPECT_THROW(
        filtering::ExtendedKalmanFilter ekf(
            x0_, P0_, nullptr, sensor_model_, Q_func_, 0.0
        ),
        std::invalid_argument
    );
}

TEST_F(ExtendedKalmanFilterTest, ConstructorThrowsOnNullSensorModel) {
    EXPECT_THROW(
        filtering::ExtendedKalmanFilter ekf(
            x0_, P0_, propagator_, nullptr, Q_func_, 0.0
        ),
        std::invalid_argument
    );
}

// ========================================
// Predict Tests
// ========================================

TEST_F(ExtendedKalmanFilterTest, PredictUpdatesState) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    Eigen::VectorXd state_before = ekf.get_state();
    
    // Predict forward
    ekf.predict(1.0);
    
    Eigen::VectorXd state_after = ekf.get_state();
    
    // State should have changed
    EXPECT_FALSE(state_after.isApprox(state_before, 1e-6));
    
    // Position should have changed (satellite is moving)
    Eigen::Vector3d pos_before = state_before.head<3>();
    Eigen::Vector3d pos_after = state_after.head<3>();
    double pos_change = (pos_after - pos_before).norm();
    
    // With velocity ~7670 m/s, after 1 second, position should change by ~7670m
    EXPECT_GT(pos_change, 7000.0);   // At least 7km
    EXPECT_LT(pos_change, 8000.0);   // But not more than 8km
    
    // Velocity should have changed (gravity is acting)
    Eigen::Vector3d vel_before = state_before.tail<3>();
    Eigen::Vector3d vel_after = state_after.tail<3>();
    double vel_change = (vel_after - vel_before).norm();
    
    // Gravity acceleration ~8.7 m/s², so Δv ~ 8.7 m/s after 1 second
    EXPECT_GT(vel_change, 5.0);    // At least 5 m/s change
    EXPECT_LT(vel_change, 15.0);   // But not more than 15 m/s
    
    // Time should have advanced
    EXPECT_DOUBLE_EQ(ekf.get_time(), 1.0);
}

TEST_F(ExtendedKalmanFilterTest, PredictUpdatesTime) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    EXPECT_DOUBLE_EQ(ekf.get_time(), 0.0);
    
    ekf.predict(1.0);
    EXPECT_DOUBLE_EQ(ekf.get_time(), 1.0);
    
    ekf.predict(0.5);
    EXPECT_DOUBLE_EQ(ekf.get_time(), 1.5);
}

TEST_F(ExtendedKalmanFilterTest, PredictIncreasesCovariance) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    Eigen::MatrixXd P_before = ekf.get_covariance();
    double trace_before = P_before.trace();
    
    // Predict forward (uncertainty should grow)
    ekf.predict(1.0);
    
    Eigen::MatrixXd P_after = ekf.get_covariance();
    double trace_after = P_after.trace();
    
    // Trace should increase (total uncertainty grows)
    EXPECT_GT(trace_after, trace_before);
}

TEST_F(ExtendedKalmanFilterTest, PredictThrowsOnNegativeTimeStep) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    EXPECT_THROW(ekf.predict(-1.0), std::invalid_argument);
    EXPECT_THROW(ekf.predict(0.0), std::invalid_argument);
}

TEST_F(ExtendedKalmanFilterTest, PredictWithGravity) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    Eigen::VectorXd state_before = ekf.get_state();
    
    std::cout << "Initial state: " << state_before.transpose() << "\n";
    std::cout << "Initial velocity: " << state_before.tail<3>().transpose() << "\n";
    std::cout << "Initial speed: " << state_before.tail<3>().norm() << "\n\n";
    
    // Predict 1 second (gravity should pull inward)
    ekf.predict(1.0);
    
    Eigen::VectorXd state_after = ekf.get_state();
    
    std::cout << "After 1s state: " << state_after.transpose() << "\n";
    std::cout << "After 1s velocity: " << state_after.tail<3>().transpose() << "\n";
    std::cout << "After 1s speed: " << state_after.tail<3>().norm() << "\n\n";
    
    Eigen::Vector3d v_before = state_before.tail<3>();
    Eigen::Vector3d v_after = state_after.tail<3>();
    
    double speed_before = v_before.norm();
    double speed_after = v_after.norm();
    
    // Speed should be similar (circular orbit)
    EXPECT_NEAR(speed_after, speed_before, 100.0);  // Within 100 m/s
    
    // Check that radial velocity component has changed (more direct test)
    // Initial position is on +x axis, so v_x should decrease (pulled toward origin)
    double vx_before = state_before(3);
    double vx_after = state_after(3);
    
    std::cout << "v_x change: " << vx_before << " -> " << vx_after << "\n";
    std::cout << "Δv_x = " << (vx_after - vx_before) << " m/s\n";
    
    // v_x should become negative (pulled toward origin)
    EXPECT_LT(vx_after, vx_before);
    EXPECT_LT(vx_after, 0.0);  // Should be negative after 1 second
    
    // The change should be reasonable (~8-9 m/s for LEO)
    double delta_vx = vx_after - vx_before;
    EXPECT_LT(delta_vx, -5.0);   // At least -5 m/s change
    EXPECT_GT(delta_vx, -15.0);  // But not more than -15 m/s
}

// ========================================
// Update Tests
// ========================================

TEST_F(ExtendedKalmanFilterTest, UpdateReducesCovariance) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    // Get covariance before update
    Eigen::MatrixXd P_before = ekf.get_covariance();
    double trace_before = P_before.trace();
    
    // Create a measurement from current state
    sensor::SensorContext ctx;
    ctx.state = ekf.get_state();
    ctx.time = 0.0;
    Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);
    
    // Create measurement with noise
    common::Measurement meas(z, sensor_model_->get_noise_covariance(), 0.0);
    
    // Update
    ekf.update(meas);
    
    // Get covariance after update
    Eigen::MatrixXd P_after = ekf.get_covariance();
    double trace_after = P_after.trace();
    
    // Trace should decrease (measurement reduces uncertainty)
    EXPECT_LT(trace_after, trace_before);
}

TEST_F(ExtendedKalmanFilterTest, UpdateWithPerfectMeasurement) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    // Get true state
    Eigen::VectorXd true_state = ekf.get_state();
    
    // Create perfect measurement (no noise)
    sensor::SensorContext ctx;
    ctx.state = true_state;
    ctx.time = 0.0;
    Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);
    
    // Very small measurement noise
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 1e-6;
    common::Measurement meas(z, R, 0.0);
    
    // Update
    ekf.update(meas);
    
    // State should remain close to true state
    Eigen::VectorXd updated_state = ekf.get_state();
    double position_error = (updated_state.head<3>() - true_state.head<3>()).norm();
    
    // Error should be very small (measurement was perfect)
    EXPECT_LT(position_error, 1.0);  // Less than 1 meter
}

TEST_F(ExtendedKalmanFilterTest, UpdateCovarianceIsSymmetric) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    // Create measurement
    sensor::SensorContext ctx;
    ctx.state = ekf.get_state();
    ctx.time = 0.0;
    Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);
    common::Measurement meas(z, sensor_model_->get_noise_covariance(), 0.0);
    
    // Update
    ekf.update(meas);
    
    // Check symmetry
    Eigen::MatrixXd P = ekf.get_covariance();
    Eigen::MatrixXd P_transpose = P.transpose();
    
    EXPECT_TRUE(P.isApprox(P_transpose, 1e-10));
}

TEST_F(ExtendedKalmanFilterTest, UpdateChangesTime) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    EXPECT_DOUBLE_EQ(ekf.get_time(), 0.0);
    
    // Create measurement at t=5.0
    sensor::SensorContext ctx;
    ctx.state = ekf.get_state();
    ctx.time = 5.0;
    Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);
    common::Measurement meas(z, sensor_model_->get_noise_covariance(), 5.0);
    
    // Update
    ekf.update(meas);
    
    // Time should be updated to measurement time
    EXPECT_DOUBLE_EQ(ekf.get_time(), 5.0);
}

// ========================================
// Predict-Update Cycle Tests
// ========================================

TEST_F(ExtendedKalmanFilterTest, PredictUpdateCycle) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    std::cout << "Initial state: " << ekf.get_state().transpose() << "\n";
    std::cout << "Initial P trace: " << ekf.get_covariance().trace() << "\n\n";
    
    // Run 10 predict-update cycles
    for (int i = 0; i < 10; ++i) {
        double t = i * 1.0;
        
        std::cout << "=== Iteration " << i << " (t=" << t << ") ===\n";
        
        // Predict
        ekf.predict(1.0);
        
        Eigen::VectorXd state_after_predict = ekf.get_state();
        double P_trace_after_predict = ekf.get_covariance().trace();
        
        std::cout << "After predict:\n";
        std::cout << "  State: " << state_after_predict.transpose() << "\n";
        std::cout << "  P trace: " << P_trace_after_predict << "\n";
        
        // Create measurement from predicted state
        sensor::SensorContext ctx;
        ctx.state = ekf.get_state();
        ctx.time = t + 1.0;
        Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);
        
        // Add some noise
        z(0) += 5.0;  // 5m range error
        
        std::cout << "  Measurement: " << z.transpose() << "\n";
        
        common::Measurement meas(z, sensor_model_->get_noise_covariance(), t + 1.0);
        
        // Update
        ekf.update(meas);
        
        Eigen::VectorXd state_after_update = ekf.get_state();
        double P_trace_after_update = ekf.get_covariance().trace();
        
        std::cout << "After update:\n";
        std::cout << "  State: " << state_after_update.transpose() << "\n";
        std::cout << "  P trace: " << P_trace_after_update << "\n\n";
        
        // Check for NaN or Inf
        if (!state_after_update.allFinite()) {
            std::cout << "ERROR: State has NaN or Inf!\n";
            std::cout << "Covariance:\n" << ekf.get_covariance() << "\n";
            FAIL() << "State diverged at iteration " << i;
        }
        
        // Check for explosion
        if (state_after_update.norm() > 1e8) {
            std::cout << "ERROR: State exploded!\n";
            std::cout << "Covariance:\n" << ekf.get_covariance() << "\n";
            FAIL() << "State exploded at iteration " << i;
        }
    }
    
    // Filter should still be running
    EXPECT_DOUBLE_EQ(ekf.get_time(), 10.0);
    
    // State should be reasonable
    Eigen::VectorXd final_state = ekf.get_state();
    std::cout << "Final state: " << final_state.transpose() << "\n";
    EXPECT_GT(final_state(0), 0.0);  // Positive x position
}

TEST_F(ExtendedKalmanFilterTest, CovarianceGrowsAndShrinks) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    double initial_trace = ekf.get_covariance().trace();
    
    // Predict (covariance should grow)
    ekf.predict(1.0);
    double after_predict_trace = ekf.get_covariance().trace();
    EXPECT_GT(after_predict_trace, initial_trace);
    
    // Update (covariance should shrink)
    sensor::SensorContext ctx;
    ctx.state = ekf.get_state();
    ctx.time = 1.0;
    Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);
    common::Measurement meas(z, sensor_model_->get_noise_covariance(), 1.0);
    
    ekf.update(meas);
    double after_update_trace = ekf.get_covariance().trace();
    EXPECT_LT(after_update_trace, after_predict_trace);
}

// ========================================
// Integration Test with Radar
// ========================================

TEST_F(ExtendedKalmanFilterTest, TrackBallisticTrajectoryWithRadar) {
    // Setup initial states
    double earth_radius = 6.371e6;
    double altitude = 400e3;
    double orbital_radius = earth_radius + altitude;
    
    Eigen::VectorXd x_true(6);
    x_true << orbital_radius, 0.0, 0.0,
              0.0, 7670.0, 0.0;
    
    Eigen::VectorXd x_filter(6);
    x_filter << orbital_radius + 1000.0, 100.0, -100.0,
                10.0, 7690.0 + 20.0, -5.0;
    
    Eigen::MatrixXd P_init = Eigen::MatrixXd::Identity(6, 6);
    P_init.block<3,3>(0,0) *= 1000.0 * 1000.0;
    P_init.block<3,3>(3,3) *= 50.0 * 50.0;
    
    filtering::ExtendedKalmanFilter ekf(
        x_filter, P_init, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    std::cout << "True initial state:   " << x_true.transpose() << "\n";
    std::cout << "Filter initial state: " << x_filter.transpose() << "\n\n";
    
    double dt = 1.0;
    int num_steps = 100;
    
    for (int i = 0; i < num_steps; ++i) {
        double t = i * dt;
        
        auto true_traj = propagator_->propagate(t, x_true, t + dt);
        x_true = true_traj.back().second;
        
        ekf.predict(dt);
        
        sensor::SensorContext ctx;
        ctx.state = x_true;
        ctx.time = t + dt;
        Eigen::VectorXd z_true = sensor_model_->compute_measurement(ctx);
        
        // Add realistic noise
        Eigen::Vector3d noise;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> range_noise(0.0, 10.0);
        std::normal_distribution<> angle_noise(0.0, 0.001);
        
        noise << range_noise(gen), angle_noise(gen), angle_noise(gen);
        Eigen::VectorXd z = z_true + noise;
        
        // Use R from sensor_model_
        common::Measurement meas(z, sensor_model_->get_noise_covariance(), t + dt);
        ekf.update(meas);
        
        if ((i + 1) % 10 == 0) {
            Eigen::VectorXd x_est = ekf.get_state();
            double pos_error = (x_est.head<3>() - x_true.head<3>()).norm();
            double vel_error = (x_est.tail<3>() - x_true.tail<3>()).norm();
            
            std::cout << "t=" << (t + dt) << "s: "
                      << "pos_error=" << pos_error << "m, "
                      << "vel_error=" << vel_error << "m/s\n";
        }
    }
    
    Eigen::VectorXd x_final = ekf.get_state();
    double position_error = (x_final.head<3>() - x_true.head<3>()).norm();
    double velocity_error = (x_final.tail<3>() - x_true.tail<3>()).norm();
    
    std::cout << "\nFinal errors:\n";
    std::cout << "  Position: " << position_error << " m\n";
    std::cout << "  Velocity: " << velocity_error << " m/s\n";
    
    EXPECT_LT(position_error, 2000.0);
    EXPECT_LT(velocity_error, 50.0);
}

// ========================================
// Reset Tests
// ========================================

TEST_F(ExtendedKalmanFilterTest, ResetRestoresInitialConditions) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    // Run filter for a while
    ekf.predict(5.0);
    
    // State and time should have changed
    EXPECT_NE(ekf.get_state()(0), x0_(0));
    EXPECT_DOUBLE_EQ(ekf.get_time(), 5.0);
    
    // Reset to new initial conditions
    Eigen::VectorXd new_x0 = Eigen::VectorXd::Zero(6);
    Eigen::MatrixXd new_P0 = Eigen::MatrixXd::Identity(6, 6);
    
    ekf.reset(new_x0, new_P0, 0.0);
    
    // State should be reset
    EXPECT_DOUBLE_EQ(ekf.get_state()(0), 0.0);
    EXPECT_DOUBLE_EQ(ekf.get_time(), 0.0);
    EXPECT_DOUBLE_EQ(ekf.get_covariance()(0,0), 1.0);
}

// ========================================
// Getter/Setter Tests
// ========================================

TEST_F(ExtendedKalmanFilterTest, SetStateUpdatesState) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    Eigen::VectorXd new_state = Eigen::VectorXd::Zero(6);
    ekf.set_state(new_state);
    
    Eigen::VectorXd retrieved_state = ekf.get_state();
    EXPECT_TRUE(retrieved_state.isApprox(new_state));
}

TEST_F(ExtendedKalmanFilterTest, SetCovarianceUpdatesCovariance) {
    filtering::ExtendedKalmanFilter ekf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0
    );
    
    Eigen::MatrixXd new_P = Eigen::MatrixXd::Identity(6, 6) * 2.0;
    ekf.set_covariance(new_P);
    
    Eigen::MatrixXd retrieved_P = ekf.get_covariance();
    EXPECT_TRUE(retrieved_P.isApprox(new_P));
}
