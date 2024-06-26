//
// Created by luc on 10/06/24.
//

#include "DescriptorPool.h"

namespace vent {
    DescriptorPool::Builder &DescriptorPool::Builder::addPoolSize(vk::DescriptorPoolSize poolSize) {
        m_poolSizes.push_back(poolSize);
        return *this;
    }

    DescriptorPool DescriptorPool::Builder::build() const {
        return {m_deviceRef, m_poolSizes, m_poolFlags, m_poolMaxSets};
    }

    DescriptorPool::DescriptorPool(const Device &device, const std::vector<vk::DescriptorPoolSize> &poolSizes, vk::DescriptorPoolCreateFlags poolFlags, uint32_t poolMaxSets)
            : m_deviceRef(device) {
        vk::DescriptorPoolCreateInfo poolInfo({}, poolMaxSets, poolSizes.size(), poolSizes.data());
        m_descriptorPool = m_deviceRef.getDevice().createDescriptorPool(poolInfo);
    }

    DescriptorPool::~DescriptorPool() {
        m_deviceRef.getDevice().resetDescriptorPool(m_descriptorPool);
        m_deviceRef.getDevice().destroyDescriptorPool(m_descriptorPool);
    }

}
