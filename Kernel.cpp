//
// Created by luc on 10/06/24.
//

#include <iostream>
#include <utility>
#include "Kernel.h"

namespace vent {
    Kernel::Kernel(const Device &device, const std::vector<vk::DescriptorSetLayoutBinding>& layout, std::string shaderData)
            : m_deviceRef(device), m_shaderData(std::move(shaderData)), m_layout(layout){
        auto compilerOptions = shaderc::CompileOptions();
        compilerOptions.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
        const auto compiled = shaderc::Compiler().CompileGlslToSpv(m_shaderData, shaderc_compute_shader, "shader.comp", compilerOptions);

        if (compiled.GetNumErrors() > 0) {
            std::cerr << compiled.GetErrorMessage();
            std::cerr << "Shader Code:\n" << m_shaderData << std::endl;
            throw std::runtime_error("Failed to compile shader");
        }

        const std::vector<uint32_t> spirv (compiled.cbegin(), compiled.cend());
        m_shaderModule = device.getDevice().createShaderModule(vk::ShaderModuleCreateInfo({}, spirv));

        m_descriptorSetLayout = device.getDevice().createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), layout));
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), m_descriptorSetLayout);
        m_pipelineLayout = device.getDevice().createPipelineLayout(pipelineLayoutCreateInfo);
        m_cache = device.getDevice().createPipelineCache(vk::PipelineCacheCreateInfo());

        vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(vk::PipelineShaderStageCreateFlags(),  // Flags
                                                                   vk::ShaderStageFlagBits::eCompute,     // Stage
                                                                   m_shaderModule,					      // Shader Module
                                                                   "main");								  // Shader Entry Point
        vk::ComputePipelineCreateInfo pipelineCreateInfo(vk::PipelineCreateFlags(),	// Flags
                                                         pipelineShaderCreateInfo,	// Shader Create Info struct
                                                         m_pipelineLayout);			// Pipeline Layout
        m_pipeline = device.getDevice().createComputePipeline(m_cache, pipelineCreateInfo).value;
    }

    Kernel::~Kernel() {
        m_deviceRef.getDevice().destroyDescriptorSetLayout(m_descriptorSetLayout);
        m_deviceRef.getDevice().destroyPipeline(m_pipeline);
        m_deviceRef.getDevice().destroyPipelineLayout(m_pipelineLayout);
        m_deviceRef.getDevice().destroyPipelineCache(m_cache);
        m_deviceRef.getDevice().destroyShaderModule(m_shaderModule);
    }

    void Kernel::addDescriptorSet(DescriptorPool& pool, const std::vector<vk::DescriptorBufferInfo>& bufferInfos) {
        m_bufferInfos.emplace_back(bufferInfos);
        vk::DescriptorSetAllocateInfo allocInfo(pool.getDescriptorPool(), 1, &m_descriptorSetLayout);
        const std::vector<vk::DescriptorSet> sets = m_deviceRef.getDevice().allocateDescriptorSets(allocInfo);
        m_sets.push_back(sets.front());

        std::vector<vk::WriteDescriptorSet> descriptorWrites(bufferInfos.size());
        for (size_t i = 0; i < bufferInfos.size(); ++i) {
            descriptorWrites[i] = vk::WriteDescriptorSet(m_sets.back(), i, 0, 1,
                                                         m_layout[i].descriptorType, nullptr,
                                                         &bufferInfos[i],nullptr);
        }

        m_deviceRef.getDevice().updateDescriptorSets(descriptorWrites, nullptr);
    }

    uint32_t Kernel::findOrAddDescriptorSet(DescriptorPool& pool, const std::vector<vk::DescriptorBufferInfo>& bufferInfos) {
        for (uint32_t i = 0; i < m_bufferInfos.size(); ++i) {
            if (m_bufferInfos[i] == bufferInfos) return i;
        }
        addDescriptorSet(pool, bufferInfos);
        return m_sets.size() - 1;
    }

    void Kernel::run(vk::CommandBuffer commandBuffer, uint32_t setIdx, uint32_t x, uint32_t y, uint32_t z) const {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, m_sets[setIdx], nullptr);
        commandBuffer.dispatch(x, y, z);
    }
}

