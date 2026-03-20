cmake_minimum_required(VERSION 3.22.1)
project(JARVIS)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_COMPILER /usr/bin/g++)


# Add compiler optimizations, warnings, and other options
# target_compile_options(jarvis_lib PUBLIC
#     -O3
#     -Wall
#     -Wextra
#     -Wpedantic
#     -fopenmp
#     -march=sapphirerapids
# )
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Library target
add_library(jarvis_lib
    src/ops.cpp
    src/token_vocab_map.cpp
    src/token_id_map.cpp
    src/tensor.cpp
    src/tokenizer.cpp
    src/transformer/transformer.cpp
    src/transformer/attention_layer.cpp
    src/transformer/ffn_layer.cpp
    src/transformer/kv_cache.cpp
    include/json.hpp
)

include_directories($ENV{HOME}/.local/include)
link_directories($ENV{HOME}/.local/lib)
target_link_libraries(jarvis_lib PUBLIC sentencepiece)

target_include_directories(jarvis_lib PUBLIC include)
target_include_directories(jarvis_lib PRIVATE include/transformer)
target_link_libraries(jarvis_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(jarvis_lib PUBLIC BLAS::BLAS)

add_executable(jarvis src/main.cpp)
target_link_libraries(jarvis PRIVATE jarvis_lib)
target_compile_options(jarvis PRIVATE -march=native -mavx2 -mavx)
# target_link_libraries(jarvis PRIVATE nlohmann_json::nlohmann_json)


# add_executable(tests tests/tests.cpp)
# target_link_libraries(tests PRIVATE jarvis_lib)