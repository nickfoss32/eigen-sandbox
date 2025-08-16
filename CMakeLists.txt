CMAKE_MINIMUM_REQUIRED(VERSION 3.23)

project(eigen_sandbox
    VERSION 0.0.1
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(eigen-sandbox main.cpp)
target_include_directories(eigen-sandbox PRIVATE include)
target_include_directories(eigen-sandbox PRIVATE src)
target_link_libraries(eigen-sandbox
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)
