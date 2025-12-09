cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(noise
    gaussian_noise.cpp
)
add_library(tracking::noise ALIAS noise)
target_include_directories(noise PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(noise PUBLIC Eigen3::Eigen)
