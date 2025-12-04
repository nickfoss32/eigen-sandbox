cmake_minimum_required(VERSION 3.15)

add_library(common
    coordinate_frame.cpp
    types.cpp
)
add_library(tracking::common ALIAS common)
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(common PUBLIC Eigen3::Eigen)
