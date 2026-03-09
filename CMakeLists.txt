cmake_minimum_required(VERSION 3.22.1)
project(JARVIS)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER /usr/bin/g++)



# Add compiler optimizations, warnings, and other options
add_compile_options(-O3 -march=native -mavx2 -fopenmp -Wall -Wextra -Wpedantic)
find_package(OpenMP REQUIRED)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Library target
add_library(jarvis_lib
    src/ops.cpp
    src/concurrent_hashmap.cpp
    src/tensor.cpp
    src/tokenizer.cpp
    src/transformer.cpp
    src/utils.cpp
)

target_include_directories(jarvis_lib PUBLIC include)
target_link_libraries(jarvis_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(jarvis src/main.cpp)
target_link_libraries(jarvis PRIVATE jarvis_lib)

add_executable(tests tests/tests.cpp)
target_link_libraries(tests PRIVATE jarvis_lib)