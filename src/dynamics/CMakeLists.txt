cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(dynamics
    ballistic3d.cpp
    gravity.cpp
)
add_library(tracking::dynamics ALIAS dynamics)
target_include_directories(dynamics 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include/dynamics
)
target_link_libraries(dynamics PUBLIC Eigen3::Eigen)
