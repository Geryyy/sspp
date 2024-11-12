cmake_minimum_required(VERSION 3.29)
project(sspp)

set(CMAKE_CXX_STANDARD 20)

# Find packages
find_package(mujoco REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)  # Find OpenMP

# Add executable
add_executable(${PROJECT_NAME} main.cpp
        Timer.h
        sspp.cpp
        sspp.h)

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
