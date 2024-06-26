cmake_minimum_required(VERSION 3.27)
project(VulkanComputePlayground)

find_package(Vulkan REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 23)
add_library(ventlib Buffer.cpp
        Buffer.h
        Device.cpp
        Device.h
        Kernel.cpp
        Kernel.h
        DescriptorPool.cpp
        DescriptorPool.h
        ComputeHandler.cpp
        ComputeHandler.h
        vent.h
        VentManager.cpp
        VentManager.h
        vent.h)
add_executable(vent main.cpp)
target_link_libraries(vent ventlib Vulkan::Vulkan shaderc_shared OpenMP::OpenMP_CXX)

add_subdirectory(Google_tests)