#include <gtest/gtest.h>

#include <dynamics/forces/constant_acceleration_force.hpp>
#include <dynamics/forces/time_window_force.hpp>

#include <memory>

namespace {

TEST(TimeWindowForceTest, AppliesWrappedForceInsideActiveWindow) {
    auto wrapped_force = std::make_shared<dynamics::ConstantAccelerationForce>(
        Eigen::Vector3d(1.5, -2.0, 0.25)
    );
    dynamics::TimeWindowForce force(wrapped_force, 10.0, 20.0);

    dynamics::ForceContext ctx;
    ctx.t = 12.0;
    ctx.position = Eigen::Vector3d::Zero();
    ctx.velocity = Eigen::Vector3d::Zero();

    const Eigen::Vector3d acceleration = force.compute_acceleration(ctx);
    EXPECT_TRUE(acceleration.isApprox(Eigen::Vector3d(1.5, -2.0, 0.25), 1e-12));

    const auto [da_dr, da_dv] = force.compute_jacobian(ctx);
    EXPECT_TRUE(da_dr.isZero(1e-12));
    EXPECT_TRUE(da_dv.isZero(1e-12));
}

TEST(TimeWindowForceTest, ReturnsZeroOutsideActiveWindow) {
    auto wrapped_force = std::make_shared<dynamics::ConstantAccelerationForce>(
        Eigen::Vector3d(-4.0, 3.0, 2.0)
    );
    dynamics::TimeWindowForce force(wrapped_force, 10.0, 20.0);

    dynamics::ForceContext ctx;
    ctx.position = Eigen::Vector3d::Ones();
    ctx.velocity = Eigen::Vector3d::Ones();

    ctx.t = 9.5;
    EXPECT_TRUE(force.compute_acceleration(ctx).isZero(1e-12));

    ctx.t = 20.0;
    EXPECT_TRUE(force.compute_acceleration(ctx).isZero(1e-12));

    const auto [da_dr, da_dv] = force.compute_jacobian(ctx);
    EXPECT_TRUE(da_dr.isZero(1e-12));
    EXPECT_TRUE(da_dv.isZero(1e-12));
}

TEST(TimeWindowForceTest, ValidatesConstructionArguments) {
    auto wrapped_force = std::make_shared<dynamics::ConstantAccelerationForce>(
        Eigen::Vector3d::UnitX()
    );

    EXPECT_THROW(
        dynamics::TimeWindowForce(nullptr, 0.0, 1.0),
        std::invalid_argument
    );
    EXPECT_THROW(
        dynamics::TimeWindowForce(wrapped_force, 5.0, 5.0),
        std::invalid_argument
    );
    EXPECT_THROW(
        dynamics::TimeWindowForce(wrapped_force, 5.0, 4.0),
        std::invalid_argument
    );
}

} // namespace
