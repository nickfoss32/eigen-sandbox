cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(filtering
    extended_kalman_filter.cpp
)
add_library(tracking::filtering ALIAS filtering)
target_include_directories(filtering PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(filtering
    PUBLIC
        tracking::common
        tracking::propagator
        tracking::sensor
        Eigen3::Eigen
)
