//
// Created by luc on 10/06/24.
//

#include "VentManager.h"

namespace vent {
    VentManager::VentManager() :
    m_descriptorPool(vent::DescriptorPool::Builder(m_device)
                                                          .addPoolSize({vk::DescriptorType::eStorageBuffer, 1000})
                                                          .setMaxSets(1000)
                                                          .build())
    {

    }
}