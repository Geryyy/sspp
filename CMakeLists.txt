cmake_minimum_required(VERSION 3.25)
project(sspp)

set(CMAKE_CXX_STANDARD 20)

# Set build type to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_program(LATEST_GCC g++-12)  # Change the version as needed
if(LATEST_GCC)
    set(CMAKE_CXX_COMPILER ${LATEST_GCC})
endif()


# Set the highest optimization level and other relevant flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native -funroll-loops -ftree-vectorize")

add_subdirectory(extern/pybind11)


## Set the Python executable using Python's sys module
#execute_process(
#        COMMAND python3 -c "import sys; print(sys.executable)"
#        RESULT_VARIABLE result
#        OUTPUT_VARIABLE python_executable
#        ERROR_VARIABLE err
#        OUTPUT_STRIP_TRAILING_WHITESPACE
#)

if(result)
    message(FATAL_ERROR "Failed to find Python executable: ${err}")
else()
    message(STATUS "Using Python executable: ${python_executable}")
    set(PYTHON_EXECUTABLE ${python_executable})
endif()


#find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
# find_package(pybind11 CONFIG REQUIRED)

# Find packages
find_package(mujoco REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)  # Find OpenMP

# Add executable
add_executable(${PROJECT_NAME} main.cpp
        Timer.h
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



#pybind11_add_module(sspp_bindings python_bindings.cpp)
#target_link_libraries(sspp_bindings PRIVATE mujoco::mujoco Eigen3::Eigen)

pybind11_add_module(_sspp python_bindings.cpp)
target_link_libraries(_sspp PRIVATE mujoco::mujoco Eigen3::Eigen OpenMP::OpenMP_CXX)

install(TARGETS _sspp DESTINATION sspp)