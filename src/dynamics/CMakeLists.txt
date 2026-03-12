cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(dynamics
    forces/atmospheric_drag.cpp
    forces/constant_acceleration_force.cpp
    forces/coordinated_turn_force.cpp
    forces/fictitious_forces.cpp
    forces/gravity.cpp
    forces/time_window_force.cpp
    point_mass_dynamics.cpp
    rigid_body_dynamics.cpp
    torques/aerodynamic_torque.cpp
    torques/gravity_gradient_torque.cpp
)
add_library(tracking::dynamics ALIAS dynamics)
target_include_directories(dynamics PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dynamics PUBLIC Eigen3::Eigen)
