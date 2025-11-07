cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(integrator
    rk4.cpp
)
add_library(tracking::integrator ALIAS integrator)
target_include_directories(integrator PUBLIC ${CMAKE_SOURCE_DIR}/include/integrator)
target_link_libraries(integrator 
    PUBLIC 
        Eigen3::Eigen
        dynamics
)
