cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(fitting
    factory.cpp
    plane_fit.cpp
)
add_library(tracking::fitting ALIAS fitting)
target_include_directories(fitting 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include/fitting
)
target_link_libraries(fitting PUBLIC Eigen3::Eigen)
