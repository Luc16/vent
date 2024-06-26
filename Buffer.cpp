//
// Created by luc on 09/06/24.
//

#include "Buffer.h"

namespace vent {
    Buffer::Buffer(const Device &device, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties): m_deviceRef(device), m_size(size) {
        m_isHostBuffer = (properties & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible;
//        if (!m_isHostBuffer) {
//            std::cout << "Creating buffer of size " << size << std::endl;
//        } else {
//            std::cout << "Creating host visible buffer of size " << size << std::endl;
//        }
        vk::BufferCreateInfo bufferCreateInfo(vk::BufferCreateFlags(), size, usage, vk::SharingMode::eExclusive);
        m_buffer = device.getDevice().createBuffer(bufferCreateInfo);

        vk::MemoryRequirements memRequirements = device.getDevice().getBufferMemoryRequirements(m_buffer);

        vk::MemoryAllocateInfo allocInfo(memRequirements.size, device.findMemoryType(memRequirements.memoryTypeBits, properties));

        m_bufferMemory = device.getDevice().allocateMemory(allocInfo);

        device.getDevice().bindBufferMemory(m_buffer, m_bufferMemory, 0);
    }

    Buffer::~Buffer() {
        m_deviceRef.getDevice().destroyBuffer(m_buffer);
        m_deviceRef.getDevice().freeMemory(m_bufferMemory);
    }

    void Buffer::map(vk::DeviceSize size) {
        if (m_mapped != nullptr) return;
        auto res = m_deviceRef.getDevice().mapMemory(m_bufferMemory, 0, size, vk::MemoryMapFlags(), &m_mapped);
        if (res != vk::Result::eSuccess || m_mapped == nullptr) {
            throw std::runtime_error("Failed to map memory");
        }
    }

    void Buffer::copyTo(void* data, vk::DeviceSize size) {
        if (size == VK_WHOLE_SIZE) size = m_size;
        memcpy(m_mapped, data, size);
    }

    void Buffer::unmap() {
        m_deviceRef.getDevice().unmapMemory(m_bufferMemory);
        m_mapped = nullptr;
    }

    void Buffer::readAll(void* data) {
        if (m_isHostBuffer) {
            map();
            memcpy((void *) data, m_mapped, (size_t) m_size);
            unmap();
            return;
        }

        Buffer stagingBuffer(m_deviceRef, m_size, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        m_deviceRef.copyBuffer(m_buffer, stagingBuffer.getBuffer(), m_size);
        stagingBuffer.readAll(data);

    }

    void Buffer::writeAll(void* data) {
        if (m_isHostBuffer) {
            map();
            copyTo(data, m_size);
            unmap();
            return;
        }


        Buffer stagingBuffer(m_deviceRef, m_size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        stagingBuffer.writeAll(data);

        m_deviceRef.copyBuffer(stagingBuffer.getBuffer(), m_buffer, m_size);
    }

}