//
// Created by luc on 10/06/24.
//

#include "ComputeHandler.h"

namespace vent {
    ComputeHandler::ComputeHandler(const Device &device): m_deviceRef(device) {
        m_fence = m_deviceRef.getDevice().createFence(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
        m_semaphore = m_deviceRef.getDevice().createSemaphore(vk::SemaphoreCreateInfo());
        m_commandBuffer = m_deviceRef.allocateCommandBuffer();
    }

    ComputeHandler::~ComputeHandler() {
            m_deviceRef.freeCommandBuffer(m_commandBuffer);
            m_deviceRef.getDevice().destroyFence(m_fence);
            m_deviceRef.getDevice().destroySemaphore(m_semaphore);
    }

    void ComputeHandler::beginComputeFrame() {
//        auto res = m_deviceRef.getDevice().waitForFences(1, &m_fence, true, UINT64_MAX);
        auto res = m_deviceRef.getDevice().resetFences(1, &m_fence);

        m_commandBuffer.reset(vk::CommandBufferResetFlags());
        m_commandBuffer.begin(vk::CommandBufferBeginInfo());
        m_isComputeFrame = true;
    }

    void ComputeHandler::submitComputeFrame() {
        m_commandBuffer.end();

        vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &m_commandBuffer);
        m_deviceRef.computeQueue().submit(submitInfo, m_fence);
        auto res = m_deviceRef.getDevice().waitForFences(1, &m_fence, true, UINT64_MAX);

        if (res != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to wait for fences");
        }
        m_isComputeFrame = false;
    }

    void ComputeHandler::computeFrame(const std::function<void(vk::CommandBuffer &)> &function) {
        beginComputeFrame();
        function(m_commandBuffer);
        submitComputeFrame();
    }

    void ComputeHandler::computeBarrier(vk::CommandBuffer commandBuffer, Buffer& buffer) {
        vk::BufferMemoryBarrier bufferBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                buffer.getBuffer(),
                0,
                buffer.getSize()
        );

        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(),
                0,
                nullptr,
                1,
                &bufferBarrier,
                0,
                nullptr
        );
    }

    vk::CommandBuffer ComputeHandler::getCommandBuffer() const {
        if (!m_isComputeFrame) {
            throw std::runtime_error("Not in compute frame");
        }
        return m_commandBuffer;
    }
}
