cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(tracking
    nearest_neighbor_associator.cpp
    track.cpp
    tracker.cpp
)
add_library(tracking::tracking ALIAS tracking)
target_include_directories(tracking PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tracking
    PUBLIC
        tracking::common
        tracking::estimation
        tracking::sensor
        Eigen3::Eigen
)
