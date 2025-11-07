cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(propagator
    propagator.cpp
)
add_library(tracking::propagator ALIAS propagator)
target_include_directories(propagator PRIVATE ${CMAKE_SOURCE_DIR}/include/propagator)
target_link_libraries(propagator
    PUBLIC
        Eigen3::Eigen
        tracking::dynamics
        tracking::integrator
        tracking::transforms
)
