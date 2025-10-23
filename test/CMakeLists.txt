cmake_minimum_required(VERSION 3.14)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)

# Add test sources (adjust if you add/remove files)
set(CT_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/test/test_transforms_coord_transforms.cpp
    # ${CMAKE_SOURCE_DIR}/test/test_coord_transforms_additional.cpp
)

add_executable(coord_transforms_tests ${CT_TEST_SOURCES})
target_include_directories(coord_transforms_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(coord_transforms_tests
    PRIVATE
        GTest::gtest_main
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        Boost::program_options
        sofa::sofa
)

# Propagate IERS_EOP_FILE macro to the test target if provided by the top-level
if(DEFINED IERS_EOP_FILE)
    target_compile_definitions(coord_transforms_tests PRIVATE IERS_EOP_FILE="${IERS_EOP_FILE}")
endif()

# Ensure tests run with working directory at the build dir (helps locating data files)
# set_tests_properties(coord_transforms_tests PROPERTIES PATH ${CMAKE_BINARY_DIR})

include(GoogleTest)
gtest_discover_tests(coord_transforms_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        FAIL_REGULAR_EXPRESSION ""
    # Optional: set DISCOVERY_TIMEOUT / FILTER etc.
)