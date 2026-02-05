cmake_minimum_required(VERSION 3.18)
project(xtensor_nanobind_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Find xtensor
find_package(xtensor REQUIRED)

# Build the extension
# NOMINSIZE: Disable -Os flag which significantly slows down xtensor operations
#            (xt::sum becomes ~2x slower with -Os due to disabled inlining)
nanobind_add_module(xtensor_nanobind_test NOMINSIZE main_nanobind.cpp)

target_include_directories(xtensor_nanobind_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(xtensor_nanobind_test PRIVATE xtensor)

# Copy to source directory for testing
add_custom_command(TARGET xtensor_nanobind_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:xtensor_nanobind_test> ${CMAKE_CURRENT_SOURCE_DIR}/
)
