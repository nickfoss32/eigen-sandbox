#include <gtest/gtest.h>

#include "dynamics/forces/constant_acceleration_force.hpp"
#include "dynamics/smooth_acceleration_point_mass_dynamics.hpp"

#include <Eigen/Dense>

#include <memory>
#include <vector>

namespace {

auto make_state(
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity,
    const Eigen::Vector3d& smooth_acceleration
) -> Eigen::VectorXd {
    Eigen::VectorXd state(9);
    state << position, velocity, smooth_acceleration;
    return state;
}

} // namespace

TEST(SmoothAccelerationPointMassDynamicsTest, StateDimensionIsNine) {
    dynamics::SmoothAccelerationPointMassDynamics dynamics({}, 5.0);
    EXPECT_EQ(dynamics.get_state_dimension(), 9);
}

TEST(SmoothAccelerationPointMassDynamicsTest, RejectsNonPositiveCorrelationTime) {
    EXPECT_THROW(
        dynamics::SmoothAccelerationPointMassDynamics({}, 0.0),
        std::invalid_argument
    );
}

TEST(SmoothAccelerationPointMassDynamicsTest, SmoothAccelerationDrivesVelocityAndRelaxesToZero) {
    dynamics::SmoothAccelerationPointMassDynamics dynamics({}, 4.0);
    const Eigen::VectorXd state = make_state(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0),
        Eigen::Vector3d(8.0, -4.0, 2.0)
    );

    const Eigen::VectorXd derivative = dynamics.compute_dynamics(0.0, state);

    EXPECT_TRUE(derivative.head<3>().isApprox(Eigen::Vector3d(4.0, 5.0, 6.0)));
    EXPECT_TRUE(derivative.segment<3>(3).isApprox(Eigen::Vector3d(8.0, -4.0, 2.0)));
    EXPECT_TRUE(derivative.tail<3>().isApprox(Eigen::Vector3d(-2.0, 1.0, -0.5)));
}

TEST(SmoothAccelerationPointMassDynamicsTest, ExternalForcesAddToSmoothAcceleration) {
    dynamics::SmoothAccelerationPointMassDynamics dynamics(
        {std::make_shared<dynamics::ConstantAccelerationForce>(Eigen::Vector3d(1.0, 2.0, 3.0))},
        10.0
    );
    const Eigen::VectorXd state = make_state(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );

    const Eigen::VectorXd derivative = dynamics.compute_dynamics(0.0, state);
    EXPECT_TRUE(derivative.segment<3>(3).isApprox(Eigen::Vector3d(5.0, 7.0, 9.0)));
}

TEST(SmoothAccelerationPointMassDynamicsTest, EquilibriumAccelerationPullsSmoothState) {
    dynamics::SmoothAccelerationPointMassDynamics dynamics(
        {},
        5.0,
        [](double t) {
            return (t < 3.0) ? Eigen::Vector3d(10.0, 0.0, 0.0)
                             : Eigen::Vector3d(0.0, 0.0, 0.0);
        }
    );
    const Eigen::VectorXd state = make_state(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d(4.0, 1.0, 0.0)
    );

    const Eigen::VectorXd derivative = dynamics.compute_dynamics(2.0, state);
    EXPECT_TRUE(derivative.tail<3>().isApprox(Eigen::Vector3d(1.2, -0.2, 0.0)));
}

TEST(SmoothAccelerationPointMassDynamicsTest, JacobianIncludesAccelerationCouplingAndDecay) {
    dynamics::SmoothAccelerationPointMassDynamics dynamics({}, 8.0);
    const Eigen::VectorXd state = make_state(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Ones(),
        Eigen::Vector3d(2.0, 3.0, 4.0)
    );

    const Eigen::MatrixXd jacobian = dynamics.compute_jacobian(0.0, state);

    EXPECT_TRUE((jacobian.block<3, 3>(0, 3).isApprox(Eigen::Matrix3d::Identity())));
    EXPECT_TRUE((jacobian.block<3, 3>(3, 6).isApprox(Eigen::Matrix3d::Identity())));
    EXPECT_TRUE(
        (jacobian.block<3, 3>(6, 6).isApprox((-1.0 / 8.0) * Eigen::Matrix3d::Identity()))
    );
}

TEST(SmoothAccelerationPointMassDynamicsTest, RejectsWrongStateDimension) {
    dynamics::SmoothAccelerationPointMassDynamics dynamics({}, 5.0);
    Eigen::VectorXd bad_state = Eigen::VectorXd::Zero(6);

    EXPECT_THROW(dynamics.compute_dynamics(0.0, bad_state), std::invalid_argument);
    EXPECT_THROW(dynamics.compute_jacobian(0.0, bad_state), std::invalid_argument);
}
