cmake_minimum_required(VERSION 3.27)
project(VulkanComputePlayground)

find_package(Vulkan REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 23)
add_executable(vent main.cpp vent.h)
target_link_libraries(vent Vulkan::Vulkan shaderc_shared OpenMP::OpenMP_CXX)

add_subdirectory(Google_tests)