//
// Created by luc on 10/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_DESCRIPTORPOOL_H
#define VULKANCOMPUTEPLAYGROUND_DESCRIPTORPOOL_H


#include "Device.h"

namespace vent {
    class DescriptorPool {
    public:
        class Builder {
        public:
            explicit Builder(const Device& device): m_deviceRef(device) {}

            Builder& addPoolSize(vk::DescriptorPoolSize poolSize);
            Builder& setFlags(vk::DescriptorPoolCreateFlags flags) { m_poolFlags = flags; return *this; }
            Builder& setMaxSets(uint32_t maxSets) { m_poolMaxSets = maxSets; return *this; }
            Builder& setMaxSetsTimesSizes(uint32_t maxSets) { m_poolMaxSets = maxSets*m_poolSizes.size(); return *this; }
            [[nodiscard]] DescriptorPool build() const;

        private:
            const Device& m_deviceRef;
            std::vector<vk::DescriptorPoolSize> m_poolSizes{};
            vk::DescriptorPoolCreateFlags m_poolFlags;
            uint32_t m_poolMaxSets = 1000;
        };

        DescriptorPool(const Device &device, const std::vector<vk::DescriptorPoolSize>& poolSizes, vk::DescriptorPoolCreateFlags poolFlags, uint32_t poolMaxSets);
        ~DescriptorPool();

        [[nodiscard]] vk::DescriptorPool getDescriptorPool() const { return m_descriptorPool; }

    private:
        const Device& m_deviceRef;

        VkDescriptorPool m_descriptorPool{};
    };
}



#endif //VULKANCOMPUTEPLAYGROUND_DESCRIPTORPOOL_H
