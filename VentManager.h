//
// Created by luc on 10/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_VENTMANAGER_H
#define VULKANCOMPUTEPLAYGROUND_VENTMANAGER_H

#include "Device.h"
#include "DescriptorPool.h"
#include "ComputeHandler.h"
#include "Kernel.h"

namespace vent {
    enum GpuRegionFlags {
        none = 0,
        keepBuffers = 1 << 1,
//        copyBuffersIn = 1 << 2,
        copyBuffersOut = 1 << 2,
    };
    inline GpuRegionFlags operator| (GpuRegionFlags f1, GpuRegionFlags f2) {
        auto v1 = static_cast<size_t>(f1);
        auto v2 = static_cast<size_t>(f2);
        return static_cast<GpuRegionFlags>(v1 | v2);
    }


    class VentManager {
    private:
        VentManager();
        ~VentManager() = default;
    public:
        static VentManager& getInstance() {
            static VentManager instance;
            return instance;
        }

        [[nodiscard]] Device& getDevice() { return m_device; }
        [[nodiscard]] DescriptorPool& getDescriptorPool() { return m_descriptorPool; }
        [[nodiscard]] ComputeHandler& getComputeHandler() { return m_computeHandler; }
        [[nodiscard]] std::unordered_map<std::string, Kernel>& getKernels() { return m_kernels; }
        [[nodiscard]] std::unordered_map<void*, Buffer>& getBuffers() { return m_gpuBuffers; }
        [[nodiscard]] std::unordered_map<size_t, Buffer>& getUniformBuffers() { return m_uniformBuffers; }
        void setGpuRegionFlags(GpuRegionFlags flags) { currentFlags = flags; }
        [[nodiscard]] GpuRegionFlags getGpuRegionFlags() const { return currentFlags; }

        VentManager(VentManager const&) = delete;
        void operator=(VentManager const&) = delete;

    private:
        Device m_device{};
        DescriptorPool m_descriptorPool;
        ComputeHandler m_computeHandler{m_device};
        std::unordered_map<std::string, Kernel> m_kernels;
        std::unordered_map<void*, Buffer> m_gpuBuffers;
        std::unordered_map<size_t, Buffer> m_uniformBuffers;
        GpuRegionFlags currentFlags = GpuRegionFlags::none;
    };


}



#endif //VULKANCOMPUTEPLAYGROUND_VENTMANAGER_H
