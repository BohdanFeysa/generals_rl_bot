cmake_minimum_required(VERSION 3.15)
project(GeneralsEngine)

# Ask Python where nanobind is installed
execute_process(
    COMMAND python3 -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_DIR}")

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Compile the batched engine into a Python module (engine.so)
nanobind_add_module(engine engine.cpp)