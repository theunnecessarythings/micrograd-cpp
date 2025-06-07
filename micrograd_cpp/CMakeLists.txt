cmake_minimum_required(VERSION 3.10)
project(micrograd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add directories for header files
include_directories(src)

# Add GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# Add subdirectories for source files
add_subdirectory(src/core)
add_subdirectory(src/nn) # Will be enabled later
# add_subdirectory(src/optim) # Will be enabled later

# Enable testing
enable_testing()
add_subdirectory(tests) # Enabled for Value tests
# add_subdirectory(examples) # Will be enabled later

message(STATUS "CMake project configured")
