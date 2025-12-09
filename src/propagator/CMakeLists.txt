cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(propagator
    analytical_propagator.cpp
    factory.cpp
    numerical_propagator.cpp
)
add_library(tracking::propagator ALIAS propagator)
target_include_directories(propagator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(propagator
    PUBLIC
        Eigen3::Eigen
        tracking::dynamics
        tracking::integrator
)
