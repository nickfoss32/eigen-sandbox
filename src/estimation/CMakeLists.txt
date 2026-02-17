cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(estimation
    imm.cpp
)
add_library(tracking::estimation ALIAS estimation)
target_include_directories(estimation PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(estimation
    PUBLIC
        tracking::common
        tracking::dynamics
        tracking::filtering
        tracking::sensor
        Eigen3::Eigen
)
