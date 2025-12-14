# CMakeLists.txt for swift_spiral_ics C++ extension

cmake_minimum_required(VERSION 3.15...3.24)
project(grid_solver_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17) # Enable C++17
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Python setup
find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")

# pybind11 setup
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11_DIR: ${pybind11_DIR}")

# Add the C++ extension module
# Removed redundant add_library call
# set_target_properties(_grid_solver_cpp PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/swift_spiral_ics/physics
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/swift_spiral_ics/physics
# )

# Set compile options for pybind11
pybind11_add_module(_grid_solver_cpp MODULE src/swift_spiral_ics/physics/grid_solver_cpp.cpp)

# Ensure output goes to correct location
# Removed set_target_properties to let setup.py handle copying
# target_link_libraries(_grid_solver_cpp PUBLIC pybind11::embed Python::Python)
# set_target_properties(_grid_solver_cpp PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/swift_spiral_ics/physics"
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/swift_spiral_ics/physics"
#     ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/swift_spiral_ics/physics"
# )
