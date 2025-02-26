cmake_minimum_required(VERSION 3.24)
project(sspp)

# Ignore Conda environment
# set(CMAKE_IGNORE_PATH $ENV{CONDA_PREFIX})
# message("CMAKE_IGNORE_PATH: ${CMAKE_IGNORE_PATH}")

set(CMAKE_CXX_STANDARD 20)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Use the desired compiler if not already set
if(NOT CMAKE_CXX_COMPILER)
    find_program(LATEST_GCC g++-12)  # Change the version as needed
    if(LATEST_GCC)
        set(CMAKE_CXX_COMPILER ${LATEST_GCC} CACHE STRING "Path to the C++ compiler" FORCE)
    endif()
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native -funroll-loops -ftree-vectorize")

# Add subdirectories
add_subdirectory(extern/pybind11)

# Find dependencies
find_package(mujoco REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/include)


# Add executable
# add_executable(${PROJECT_NAME} src/main.cpp include/Timer.h include/sspp.h)
add_executable(${PROJECT_NAME} src/main_tsp.cpp include/Timer.h include/tsp.h include/utility.h)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC mujoco::mujoco
    PUBLIC Threads::Threads
    PUBLIC Eigen3::Eigen
    PUBLIC glfw
    PUBLIC OpenMP::OpenMP_CXX
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python bindings
pybind11_add_module(_sspp src/python_bindings.cpp)
target_link_libraries(_sspp PRIVATE mujoco::mujoco Eigen3::Eigen OpenMP::OpenMP_CXX)

install(TARGETS _sspp DESTINATION sspp)
