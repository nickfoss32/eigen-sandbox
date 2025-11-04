cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(transforms
    coord_transforms.cpp
)
add_library(tracking::transforms ALIAS transforms)
target_include_directories(transforms PRIVATE ${CMAKE_SOURCE_DIR}/include/transforms)
# target_include_directories(transforms PUBLIC ${SOFA_INCLUDE_DIR})
target_link_libraries(transforms
    PUBLIC
        Eigen3::Eigen
        sofa::sofa
)
