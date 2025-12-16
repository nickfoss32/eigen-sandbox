cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(dynamics
    aerodynamic_torque.cpp
    atmospheric_drag.cpp
    fictitious_forces.cpp
    gravity.cpp
    gravity_gradient_torque.cpp
    point_mass_dynamics.cpp
    rigid_body_dynamics.cpp
)
add_library(tracking::dynamics ALIAS dynamics)
target_include_directories(dynamics PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dynamics PUBLIC Eigen3::Eigen)
