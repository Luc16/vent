//
// Created by luc on 10/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_COMPUTEHANDLER_H
#define VULKANCOMPUTEPLAYGROUND_COMPUTEHANDLER_H

#include "Device.h"
#include "Buffer.h"

namespace vent {
    class ComputeHandler {
    public:
        explicit ComputeHandler(const Device &device);
        ~ComputeHandler();
        ComputeHandler(const ComputeHandler &) = delete;
        ComputeHandler &operator=(const ComputeHandler &) = delete;

        void computeFrame(const std::function<void(vk::CommandBuffer &)> &function);
        void beginComputeFrame();
        void submitComputeFrame();
        static void computeBarrier(vk::CommandBuffer commandBuffer, Buffer& buffer);
        [[nodiscard]] vk::CommandBuffer getCommandBuffer() const;
        [[nodiscard]] bool isComputeFrame() const { return m_isComputeFrame; };

    private:
        bool m_isComputeFrame = false;
        const Device &m_deviceRef;

        vk::Fence m_fence;
        vk::Semaphore m_semaphore;

        vk::CommandBuffer m_commandBuffer{};
    };
}



#endif //VULKANCOMPUTEPLAYGROUND_COMPUTEHANDLER_H
