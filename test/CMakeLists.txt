cmake_minimum_required(VERSION 3.14)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

include(GoogleTest)

# Common tests
add_executable(common_tests
    test_common_coordinate_frame.cpp
    test_common_types.cpp
)
target_link_libraries(common_tests
    PRIVATE
        GTest::gtest_main
        tracking::common
)
gtest_discover_tests(common_tests)

# Dynamics tests
add_executable(dynamics_tests
    test_dynamics_fictitious_forces.cpp
    test_dynamics_point_mass_dynamics.cpp
)
target_link_libraries(dynamics_tests
    PRIVATE
        GTest::gtest_main
        tracking::dynamics
)
gtest_discover_tests(dynamics_tests)

# Filtering tests
add_executable(filtering_tests
    test_filtering_extended_kalman_filter.cpp
)
target_link_libraries(filtering_tests
    PRIVATE
        GTest::gtest_main
        tracking::filtering
        tracking::integrator
        tracking::propagator
        tracking::sensor
        tracking::dynamics
)
gtest_discover_tests(filtering_tests)

# Integrator tests
add_executable(integrator_tests
    test_integrator_rk4.cpp
)
target_link_libraries(integrator_tests
    PRIVATE
        GTest::gtest_main
        tracking::integrator
)
gtest_discover_tests(integrator_tests)

# Noise tests
add_executable(noise_tests
    test_noise_gaussian_noise.cpp
)
target_link_libraries(noise_tests
    PRIVATE
        GTest::gtest_main
        tracking::noise
)
gtest_discover_tests(noise_tests)

# Propagator tests
add_executable(propagator_tests
    test_propagator_analytical_propagator.cpp
    test_propagator_factory.cpp
    test_propagator_numerical_propagator.cpp
)
target_link_libraries(propagator_tests
    PRIVATE
        GTest::gtest_main
        tracking::propagator
)
gtest_discover_tests(propagator_tests)

# Sensor tests
add_executable(sensor_tests
    test_sensor_radar_sensor_model.cpp
)
target_link_libraries(sensor_tests
    PRIVATE
        GTest::gtest_main
        tracking::sensor
)
gtest_discover_tests(sensor_tests)

# Transforms Tests
add_executable(coord_transforms_tests
    test_transforms_coord_transforms.cpp
)
target_link_libraries(coord_transforms_tests
    PRIVATE
        GTest::gtest_main
        tracking::transforms
)
target_compile_definitions(coord_transforms_tests PRIVATE IERS_EOP_FILE="${IERS_EOP_FILE}")
gtest_discover_tests(coord_transforms_tests)