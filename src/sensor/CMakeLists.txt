cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 REQUIRED)

add_library(sensor
    radar_sensor_model.cpp
)
add_library(tracking::sensor ALIAS sensor)
target_include_directories(sensor PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sensor 
    PUBLIC 
        Eigen3::Eigen
)
