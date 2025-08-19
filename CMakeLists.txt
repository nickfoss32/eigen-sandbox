CMAKE_MINIMUM_REQUIRED(VERSION 3.23)

project(eigen_sandbox
    VERSION 0.0.1
    LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)

add_executable(propagate-track apps/propagate_track.cpp)
target_include_directories(propagate-track PRIVATE include)
target_link_libraries(propagate-track
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        Boost::program_options
)

add_executable(generate-data apps/generate_data.cpp)
target_include_directories(generate-data PRIVATE include)
target_link_libraries(generate-data
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        Boost::program_options
)
