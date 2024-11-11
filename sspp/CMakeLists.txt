cmake_minimum_required(VERSION 3.29)
project(sspp)

set(CMAKE_CXX_STANDARD 20)

# Find packages
find_package(mujoco REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Link libraries
link_libraries(mujoco::mujoco Eigen3::Eigen)

# Add executable
add_executable(${PROJECT_NAME} main.cpp
        Timer.h
        sspp.cpp
        sspp.h)
target_link_libraries(${PROJECT_NAME} mujoco::mujoco Threads::Threads Eigen3::Eigen)