cmake_minimum_required(VERSION 3.20)
project(sspp)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Use the desired compiler if not already set
if(NOT CMAKE_CXX_COMPILER)
    find_program(LATEST_GCC g++-12)  # Change version if necessary
    if(LATEST_GCC)
        set(CMAKE_CXX_COMPILER ${LATEST_GCC} CACHE STRING "Path to the C++ compiler" FORCE)
    endif()
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native -funroll-loops -ftree-vectorize")

#set( CMAKE_VERBOSE_MAKEFILE on )

#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0")

# Add profiling flags for gperftools (heap and CPU profiling)
if(ENABLE_PROFILE)
    # Link against tcmalloc for memory profiling
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TCMALLOC -ltcmalloc")

    # Enable heap profiling (you can set the HEAPPROFILE environment var at runtime)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTCMALLOC_HEAP_PROFILER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTCMALLOC_PROFILE_PATH='/tmp/heapprof'")  # Optional: specify path

    # Link against the profiler for CPU profiling
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_PROFILER_ENABLED -lprofiler")

    # Enable CPU profiling (set the CPUPROFILE environment var at runtime)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTCMALLOC_CPU_PROFILER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTCMALLOC_PROFILE_PATH='/tmp/cpuprofile'")  # Optional: specify path
endif()

# Add subdirectories
add_subdirectory(extern/pybind11)

# Find dependencies
find_package(mujoco REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Add executable
add_executable(tsp_standalone src/main_tsp.cpp include/Timer.h include/tsp.h include/utility.h
        include/Gradient.h
        include/visu.h
        src/visu.cpp
        include/ui.h
        src/ui.cpp include/Collision.h)

# Link libraries
target_link_libraries(tsp_standalone
        PUBLIC mujoco::mujoco
        PUBLIC Threads::Threads
        PUBLIC Eigen3::Eigen
        PUBLIC glfw
        PUBLIC OpenMP::OpenMP_CXX)

# Link gperftools (for memory and CPU profiling)
if(GPERFTOOLS_LIBRARY)
    target_link_libraries(tsp_standalone PUBLIC -ltcmalloc -lprofiler)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(tsp_standalone PUBLIC OpenMP::OpenMP_CXX)
endif()

# SampledSplinePathPlanner bindings
pybind11_add_module(_sspp src/sspp_bindings.cpp)
target_link_libraries(_sspp PRIVATE mujoco::mujoco Eigen3::Eigen OpenMP::OpenMP_CXX)

install(TARGETS _sspp DESTINATION sspp)

# TaskSpacePlanner bindings
#pybind11_add_module(_tsp src/tsp_bindings.cpp)
#target_link_libraries(_tsp PRIVATE mujoco::mujoco Eigen3::Eigen OpenMP::OpenMP_CXX)

#install(TARGETS _tsp DESTINATION sspp)


# build library

add_library(sspp STATIC
    src/visu.cpp
    src/ui.cpp
    include/visu.h
    include/ui.h
    include/utility.h
    include/Gradient.h
    include/Collision.h
    include/Timer.h
    include/tsp.h
)

target_include_directories(sspp
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(sspp
    PUBLIC mujoco::mujoco
    PUBLIC Eigen3::Eigen
    PUBLIC OpenMP::OpenMP_CXX
)


# Export the CMake package config
include(CMakePackageConfigHelpers)

# Create the targets file
install(TARGETS sspp
        EXPORT ssppTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)

install(DIRECTORY include/
        DESTINATION include)

# Export targets and config
install(EXPORT ssppTargets
        FILE ssppTargets.cmake
        NAMESPACE sspp::
        DESTINATION lib/cmake/sspp)

# Generate and install the config file
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ssppConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ssppConfig.cmake"
  INSTALL_DESTINATION lib/cmake/sspp
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ssppConfig.cmake"
  DESTINATION lib/cmake/sspp
)
