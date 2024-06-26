//
// Created by luc on 09/06/24.
//

#include <iostream>
#include "Device.h"

namespace vent {
    Device::Device() {
        vk::ApplicationInfo appInfo{
                "VulkanCompute",	// Application Name
                1,					// Application Version
                nullptr,			// Engine Name or nullptr
                0,					// Engine Version
                VK_API_VERSION_1_3  // Vulkan API version
        };

        const std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
        vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(),	// Flags
                                                  &appInfo,						// Application Info
                                                  layers.size(),				// layers count
                                                  layers.data());				// Layers
        m_instance = vk::createInstance(instanceCreateInfo);

        m_physicalDevice = m_instance.enumeratePhysicalDevices()[1];
        auto p = m_physicalDevice.getProperties();
//        std::cout << "Device Name: " << p.deviceName << std::endl;


        auto queueFamilyProps = m_physicalDevice.getQueueFamilyProperties();
        auto propIt = std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties& prop)
        {
            return prop.queueFlags & vk::QueueFlagBits::eCompute;
        });
        const uint32_t queueFamilyIndex = std::distance(queueFamilyProps.begin(), propIt);

        // Just to avoid a warning from the Vulkan Validation Layer
        const float queuePriority = 1.0f;
        const vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);
        m_device = m_physicalDevice.createDevice(vk::DeviceCreateInfo({}, deviceQueueCreateInfo));
        m_computeQueue = m_device.getQueue(queueFamilyIndex, 0);

        vk::CommandPoolCreateInfo poolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), queueFamilyIndex);
        m_commandPool = m_device.createCommandPool(poolCreateInfo);

    }


    Device::~Device() {
        m_device.resetCommandPool(m_commandPool, vk::CommandPoolResetFlags());
        m_device.waitIdle();
        m_device.destroyCommandPool(m_commandPool);
        m_device.destroy();
        m_instance.destroy();
    }

    uint32_t Device::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = m_physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

    void Device::executeSingleCommand(const std::function<void(vk::CommandBuffer &)> &function) const {
        vk::CommandBuffer commandBuffer = allocateCommandBuffer();

        commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

        function(commandBuffer);

        commandBuffer.end();

        vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
        m_computeQueue.submit(submitInfo, nullptr);
        m_computeQueue.waitIdle();
        m_device.freeCommandBuffers(m_commandPool, commandBuffer);
    }

    void Device::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const {
        executeSingleCommand([&](vk::CommandBuffer& commandBuffer) {
            vk::BufferCopy copyRegion(0, 0, size);
            commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
        });
    }

    vk::CommandBuffer Device::allocateCommandBuffer(vk::CommandBufferLevel level) const {
        vk::CommandBufferAllocateInfo allocInfo(m_commandPool, level, 1);
        return m_device.allocateCommandBuffers(allocInfo).front();
    }
}