//
// Created by luc on 09/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_DEVICE_H
#define VULKANCOMPUTEPLAYGROUND_DEVICE_H

#include <vulkan/vulkan.hpp>


namespace vent {
    class Device {
    public:
        Device();
        ~Device();

        [[nodiscard]] vk::Instance instance() const { return m_instance; }
        [[nodiscard]] vk::PhysicalDevice physicalDevice() const { return m_physicalDevice; }
        [[nodiscard]] vk::Device getDevice() const { return m_device; }
        [[nodiscard]] vk::Queue computeQueue() const { return m_computeQueue; }

        [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

        void freeCommandBuffer(vk::CommandBuffer commandBuffer) const {m_device.freeCommandBuffers(m_commandPool, 1, &commandBuffer);}
        void executeSingleCommand(const std::function<void(vk::CommandBuffer&)>& function) const;
        void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const;
        [[nodiscard]] vk::CommandBuffer allocateCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;



    private:

        vk::Instance m_instance;
        vk::PhysicalDevice m_physicalDevice;
        vk::Device m_device;
        vk::Queue m_computeQueue;
        vk::CommandPool m_commandPool;

    };
}



#endif //VULKANCOMPUTEPLAYGROUND_DEVICE_H
