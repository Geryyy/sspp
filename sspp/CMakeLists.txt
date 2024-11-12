cmake_minimum_required(VERSION 3.29)
project(sspp)

set(CMAKE_CXX_STANDARD 20)

# Set build type to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set the highest optimization level and other relevant flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native -funroll-loops -ftree-vectorize")

# Find packages
find_package(mujoco REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)  # Find OpenMP

# Add executable
add_executable(${PROJECT_NAME} main.cpp
        Timer.h
        sspp.h
        dspp.h)

# Link libraries with keyword mode
target_link_libraries(${PROJECT_NAME}
        PUBLIC mujoco::mujoco
        PUBLIC Threads::Threads
        PUBLIC Eigen3::Eigen
)

# Link OpenMP libraries if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Optionally: Set the newest available compiler if required
# For example, if you want to use the latest GCC compiler
find_program(LATEST_GCC g++-12)  # Change the version as needed
if(LATEST_GCC)
    set(CMAKE_CXX_COMPILER ${LATEST_GCC})
endif()
