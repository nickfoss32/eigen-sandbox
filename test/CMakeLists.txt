cmake_minimum_required(VERSION 3.14)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(coord_transforms_tests
    test_transforms_coord_transforms.cpp
)
target_include_directories(coord_transforms_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(coord_transforms_tests
    PRIVATE
        GTest::gtest_main
        tracking::transforms
)
target_compile_definitions(coord_transforms_tests PRIVATE IERS_EOP_FILE="${IERS_EOP_FILE}")

include(GoogleTest)
gtest_discover_tests(coord_transforms_tests)
