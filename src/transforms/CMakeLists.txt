cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(transforms
    coord_transforms.cpp
)
add_library(tracking::transforms ALIAS transforms)
target_include_directories(transforms PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(transforms
    PUBLIC
        Eigen3::Eigen
        sofa::sofa
)
