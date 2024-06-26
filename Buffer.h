//
// Created by luc on 09/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_BUFFER_H
#define VULKANCOMPUTEPLAYGROUND_BUFFER_H

#include <iostream>
#include "Device.h"

namespace vent {

    class Buffer {
    public:
        Buffer(const Device& device, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);
        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;
        ~Buffer();

        void map(vk::DeviceSize size = VK_WHOLE_SIZE);
        void copyTo(void* data, vk::DeviceSize size = VK_WHOLE_SIZE);
        void unmap();

        void swap(Buffer& other) {
            std::swap(m_isHostBuffer, other.m_isHostBuffer);
            std::swap(m_changed, other.m_changed);
            std::swap(m_buffer, other.m_buffer);
            std::swap(m_bufferMemory, other.m_bufferMemory);
            std::swap(m_size, other.m_size);
            std::swap(m_mapped, other.m_mapped);
        }

        [[nodiscard]] vk::Buffer getBuffer() const { return m_buffer; }
        [[nodiscard]] vk::DeviceMemory getMemory() const { return m_bufferMemory; }
        [[nodiscard]] vk::DeviceSize getSize() const { return m_size; }
        [[nodiscard]] vk::DescriptorBufferInfo descriptorInfo(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) const
        { return {m_buffer, offset, size}; }
        [[nodiscard]] bool changed() const { return m_changed; }
        void setAsChanged() { m_changed = true; }

        void readAll(void* data);
        void writeAll(void* data);

        template<typename T>
        void write(T& data);

        template<typename InputIt>
        void write(InputIt first, InputIt last);

        template<typename T>
        void read(T& data);

        template<typename InputIt>
        void read(InputIt first, InputIt last);

    private:

        const Device& m_deviceRef;
        bool m_isHostBuffer = false;
        bool m_changed = false;
        vk::Buffer m_buffer;
        vk::DeviceMemory m_bufferMemory;
        vk::DeviceSize m_size;
        void* m_mapped = nullptr;
    };


    template<typename T>
    void Buffer::write(T& data) {
        if (m_isHostBuffer) {
            map();
            copyTo((void*) &data, sizeof(T));
            unmap();
            return;
        }

        size_t size = sizeof(data);
        if (size > m_size) {
            throw std::runtime_error("Buffer too small to receive data, struct has " +
                                     std::to_string(size) +
                                     " bytes, buffer has " +
                                     std::to_string(m_size) +
                                     " bytes");        }
        Buffer stagingBuffer(m_deviceRef, size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        stagingBuffer.write(data);

        m_deviceRef.copyBuffer(stagingBuffer.getBuffer(), m_buffer, size);
    }

    template<typename InputIt>
    void Buffer::write(InputIt first, InputIt last) {
        size_t size = std::distance(first, last)*sizeof(*first);
        if (m_isHostBuffer) {
            map();
            auto pointer = static_cast<decltype(&(*first))>(m_mapped);
            #pragma omp parallel for
            for (uint32_t i = 0; i < std::distance(first, last); i++) {
                pointer[i] = first[i];
            }
            std::copy(first, last, static_cast<decltype(&(*first))>(m_mapped));
            unmap();
            return;
        }

        if (size > m_size) {
            throw std::runtime_error("Buffer too small to receive data, vector has " +
                                     std::to_string(size) +
                                     " bytes, buffer has " +
                                     std::to_string(m_size) +
                                     " bytes");        }
        Buffer stagingBuffer(m_deviceRef, size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        stagingBuffer.write(first, last);

        m_deviceRef.copyBuffer(stagingBuffer.getBuffer(), m_buffer, size);
    }

    template<typename T>
    void Buffer::read(T &data) {
        if (m_isHostBuffer) {
            map();
            memcpy((void *) &data, m_mapped, (size_t) m_size);
            unmap();
            return;
        }

        size_t size = std::min(sizeof(T), m_size);
        Buffer stagingBuffer(m_deviceRef, size, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        m_deviceRef.copyBuffer(m_buffer, stagingBuffer.getBuffer(), size);
        stagingBuffer.read(data);

    }

    template<typename InputIt>
    void Buffer::read(InputIt first, InputIt last) {
        size_t size = std::distance(first, last)*sizeof(*first);

        if (m_isHostBuffer) {
            map();
            auto pointer = static_cast<decltype(&(*first))>(m_mapped);
            #pragma omp parallel for
            for (uint32_t i = 0; i < std::distance(first, last); i++) {
                first[i] = pointer[i];
            }
            unmap();
            return;
        }

        if (m_size > size) {
            throw std::runtime_error("Vector too small to receive data, vector has " +
                                     std::to_string(size) +
                                     " bytes, buffer has " +
                                     std::to_string(m_size) +
                                     " bytes");
        }
        Buffer stagingBuffer(m_deviceRef, size, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        m_deviceRef.copyBuffer(m_buffer, stagingBuffer.getBuffer(), m_size);
        stagingBuffer.read(first, last);

    }
}




#endif //VULKANCOMPUTEPLAYGROUND_BUFFER_H
