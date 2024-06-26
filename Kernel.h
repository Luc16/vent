//
// Created by luc on 10/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_KERNEL_H
#define VULKANCOMPUTEPLAYGROUND_KERNEL_H

#include "Device.h"
#include "DescriptorPool.h"
#include <shaderc/shaderc.hpp>

namespace vent {
    class Kernel {
    public:
        Kernel(const Device &device, const std::vector<vk::DescriptorSetLayoutBinding>& layout, std::string  shaderData);
        Kernel(Kernel& other) = delete;
        Kernel operator=(Kernel& other) = delete;
        ~Kernel();

        void addDescriptorSet(DescriptorPool& pool, const std::vector<vk::DescriptorBufferInfo>& bufferInfos);
        uint32_t findOrAddDescriptorSet(DescriptorPool& pool, const std::vector<vk::DescriptorBufferInfo>& bufferInfos);
        void run(vk::CommandBuffer commandBuffer, uint32_t setIdx, uint32_t x, uint32_t y, uint32_t z) const;

        [[nodiscard]] vk::Pipeline getPipeline() const { return m_pipeline; }

    private:
        const Device& m_deviceRef;
        std::string m_shaderData;
        vk::Pipeline m_pipeline;
        vk::PipelineLayout m_pipelineLayout;
        vk::PipelineCache m_cache;
        vk::ShaderModule m_shaderModule;
        vk::DescriptorSetLayout m_descriptorSetLayout;
        std::vector<vk::DescriptorSet> m_sets;
        std::vector<std::vector<vk::DescriptorBufferInfo>> m_bufferInfos;
        std::vector<vk::DescriptorSetLayoutBinding> m_layout;
    };
}

#endif //VULKANCOMPUTEPLAYGROUND_KERNEL_H
