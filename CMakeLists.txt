cmake_minimum_required(VERSION 3.20)
project(sspp LANGUAGES CXX)

# --- C++ standard & build type ------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Compiler selection (optional) -------------------------------------------
if(NOT CMAKE_CXX_COMPILER)
  find_program(LATEST_GCC g++-12)
  if(LATEST_GCC)
    set(CMAKE_CXX_COMPILER ${LATEST_GCC} CACHE STRING "C++ compiler" FORCE)
  endif()
endif()

# --- Optimizations ------------------------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native -funroll-loops -ftree-vectorize")

# --- Options -----------------------------------------------------------------
option(DEBUG_SINGLE_THREAD "Disable OpenMP for debugging" OFF)
option(ENABLE_PROFILE "Enable gperftools profiling" OFF)

# --- Dependencies -------------------------------------------------------------
add_subdirectory(extern/pybind11) # pybind11

find_package(mujoco REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# glfw is provided by MuJoCo prebuilt, but if you need a separate find, add it.
# find_package(glfw3 REQUIRED)

# --- Include dirs -------------------------------------------------------------
# Put your headers (including the new modular ones) under include/
# e.g., include/tsp_adapter_task_space_planner.h, tsp_*.h
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Small helper to link OpenMP conditionally -------------------------------
function(link_openmp_if target)
  if(NOT DEBUG_SINGLE_THREAD)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
      target_compile_definitions(${target} PUBLIC -D_USE_OPENMP)
    endif()
  else()
    target_compile_definitions(${target} PUBLIC -DDEBUG_SINGLE_THREAD=1)
  endif()
endfunction()

# --- Sources (only .cpp files; headers are included automatically) -----------
set(SSPP_COMMON_SRCS
    src/visu.cpp
    src/ui.cpp
)

set(SSPP_COMMON_INCLUDES
    include/visu.h
    include/ui.h
    include/utility.h
#    include/Gradient.h
    include/Collision.h
    include/Timer.h
    # modular planner headers (adapter + modules)
#    include/tsp_adapter_task_space_planner.h
    include/sspp/tsp_types.h
    include/sspp/tsp_sampler.h
    include/sspp/tsp_evaluator.h
    include/sspp/tsp_elites.h
    include/sspp/tsp_distribution.h
    include/sspp/tsp_collision_world.h
    include/sspp/tsp_path_model.h
    include/sspp/tsp_planner.h
)

# --- Executables --------------------------------------------------------------
add_executable(tsp_standalone
    src/main_tsp.cpp
    ${SSPP_COMMON_SRCS}
    ${SSPP_COMMON_INCLUDES}
)
target_link_libraries(tsp_standalone
    PUBLIC mujoco::mujoco
    PUBLIC Threads::Threads
    PUBLIC Eigen3::Eigen
    PUBLIC glfw
)
link_openmp_if(tsp_standalone)

add_executable(tsp_benchmark
    #src/main_tsp_benchmark.cpp
    src/main_icra_benchmark.cpp
#    ${SSPP_COMMON_SRCS}
    ${SSPP_COMMON_INCLUDES}
)
target_link_libraries(tsp_benchmark
    PUBLIC mujoco::mujoco
    PUBLIC Threads::Threads
    PUBLIC Eigen3::Eigen
    PUBLIC glfw
)
link_openmp_if(tsp_benchmark)

# --- Profiling (optional) ----------------------------------------------------
if(ENABLE_PROFILE)
  # Expect gperftools in system linker path; adjust if needed with find_library
  target_link_libraries(tsp_standalone PUBLIC profiler tcmalloc)
  target_link_libraries(tsp_benchmark  PUBLIC profiler tcmalloc)
  target_compile_definitions(tsp_standalone PUBLIC USE_TCMALLOC CPU_PROFILER_ENABLED)
  target_compile_definitions(tsp_benchmark  PUBLIC USE_TCMALLOC CPU_PROFILER_ENABLED)
endif()

# --- Python bindings ----------------------------------------------------------
pybind11_add_module(_sspp src/sspp_bindings.cpp)
target_link_libraries(_sspp PRIVATE mujoco::mujoco Eigen3::Eigen)
link_openmp_if(_sspp)
target_include_directories(_sspp PRIVATE ${CMAKE_SOURCE_DIR}/include)

pybind11_add_module(_tsp src/tsp_bindings.cpp)
target_link_libraries(_tsp PRIVATE mujoco::mujoco Eigen3::Eigen)
link_openmp_if(_tsp)
target_include_directories(_tsp PRIVATE ${CMAKE_SOURCE_DIR}/include)

install(TARGETS _sspp _tsp DESTINATION sspp)

# --- Build static lib (optional; remove if unused) ---------------------------
add_library(sspp STATIC
    ${SSPP_COMMON_SRCS}
    ${SSPP_COMMON_INCLUDES}
)
target_include_directories(sspp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(sspp
  PUBLIC mujoco::mujoco
  PUBLIC Eigen3::Eigen
)
link_openmp_if(sspp)

# --- Install / export ---------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS sspp
        EXPORT ssppTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT ssppTargets
        FILE ssppTargets.cmake
        NAMESPACE sspp::
        DESTINATION lib/cmake/sspp)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ssppConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ssppConfig.cmake"
  INSTALL_DESTINATION lib/cmake/sspp
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ssppConfig.cmake"
        DESTINATION lib/cmake/sspp)
