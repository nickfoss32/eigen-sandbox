cmake_minimum_required(VERSION 3.23)

project(eigen_sandbox
    VERSION 0.0.1
    LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to control building tests
option(BUILD_TESTS "Build tests" ON)

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# find dependencies
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)

# Download IERS EOP data
find_package(IERS REQUIRED)
message(STATUS "IERS EOP file: ${IERS_EOP_FILE}")

# build SOFA
set(SOFA_SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/sofa/c/src)
file(GLOB SOFA_SOURCES ${SOFA_SOURCE_DIR}/*.c)
add_library(sofa STATIC ${SOFA_SOURCES})
add_library(sofa::sofa ALIAS sofa)
target_include_directories(sofa PUBLIC ${SOFA_SOURCE_DIR})
set_target_properties(sofa PROPERTIES LINKER_LANGUAGE C)

# Also expose SOFA include dir as a variable so existing code using ${SOFA_INCLUDE_DIR} continues to work
set(SOFA_INCLUDE_DIR ${SOFA_SOURCE_DIR} CACHE INTERNAL "SOFA include directory")

# build apps
add_executable(propagate-track apps/propagate_track.cpp)
target_compile_definitions(propagate-track PRIVATE IERS_EOP_FILE="${IERS_EOP_FILE}")
target_include_directories(propagate-track PRIVATE include ${SOFA_INCLUDE_DIR})
target_link_libraries(propagate-track
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        Boost::program_options
        sofa::sofa
)

add_executable(generate-data apps/generate_data.cpp)
target_compile_definitions(generate-data PRIVATE IERS_EOP_FILE="${IERS_EOP_FILE}")
target_include_directories(generate-data PRIVATE include ${SOFA_INCLUDE_DIR})
target_link_libraries(generate-data
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        Boost::program_options
        sofa::sofa
)

# Optionally build tests
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
