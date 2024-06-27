//
// Created by luc on 11/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_VENT_H
#define VULKANCOMPUTEPLAYGROUND_VENT_H

#include <vector>
#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp>

namespace vent {
    class Device {
    public:
        Device() {
            vk::ApplicationInfo appInfo{
                    "VulkanCompute",	// Application Name
                    1,					// Application Version
                    nullptr,			// Engine Name or nullptr
                    0,					// Engine Version
                    VK_API_VERSION_1_3  // Vulkan API version
            };

            const std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
            vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(),	// Flags
                                                      &appInfo,						// Application Info
                                                      layers.size(),				// layers count
                                                      layers.data());				// Layers
            m_instance = vk::createInstance(instanceCreateInfo);

            m_physicalDevice = m_instance.enumeratePhysicalDevices()[1];
            auto p = m_physicalDevice.getProperties();
//        std::cout << "Device Name: " << p.deviceName << std::endl;


            auto queueFamilyProps = m_physicalDevice.getQueueFamilyProperties();
            auto propIt = std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties& prop)
            {
                return prop.queueFlags & vk::QueueFlagBits::eCompute;
            });
            const uint32_t queueFamilyIndex = std::distance(queueFamilyProps.begin(), propIt);

            // Just to avoid a warning from the Vulkan Validation Layer
            const float queuePriority = 1.0f;
            const vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);
            m_device = m_physicalDevice.createDevice(vk::DeviceCreateInfo({}, deviceQueueCreateInfo));
            m_computeQueue = m_device.getQueue(queueFamilyIndex, 0);

            vk::CommandPoolCreateInfo poolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), queueFamilyIndex);
            m_commandPool = m_device.createCommandPool(poolCreateInfo);

        }

        ~Device() {
            m_device.resetCommandPool(m_commandPool, vk::CommandPoolResetFlags());
            m_device.waitIdle();
            m_device.destroyCommandPool(m_commandPool);
            m_device.destroy();
            m_instance.destroy();
        }

        [[nodiscard]] vk::Instance instance() const { return m_instance; }
        [[nodiscard]] vk::PhysicalDevice physicalDevice() const { return m_physicalDevice; }
        [[nodiscard]] vk::Device getDevice() const { return m_device; }
        [[nodiscard]] vk::Queue computeQueue() const { return m_computeQueue; }

        [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
            vk::PhysicalDeviceMemoryProperties memProperties = m_physicalDevice.getMemoryProperties();

            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }

            throw std::runtime_error("Failed to find suitable memory type!");
        }

        void freeCommandBuffer(vk::CommandBuffer commandBuffer) const {m_device.freeCommandBuffers(m_commandPool, 1, &commandBuffer);}
        void executeSingleCommand(const std::function<void(vk::CommandBuffer&)>& function) const {
            vk::CommandBuffer commandBuffer = allocateCommandBuffer();

            commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

            function(commandBuffer);

            commandBuffer.end();

            vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
            m_computeQueue.submit(submitInfo, nullptr);
            m_computeQueue.waitIdle();
            m_device.freeCommandBuffers(m_commandPool, commandBuffer);
        }
        void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const {
            executeSingleCommand([&](vk::CommandBuffer& commandBuffer) {
                vk::BufferCopy copyRegion(0, 0, size);
                commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
            });
        }
        [[nodiscard]] vk::CommandBuffer allocateCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const {
            vk::CommandBufferAllocateInfo allocInfo(m_commandPool, level, 1);
            return m_device.allocateCommandBuffers(allocInfo).front();
        }


    private:

        vk::Instance m_instance;
        vk::PhysicalDevice m_physicalDevice;
        vk::Device m_device;
        vk::Queue m_computeQueue;
        vk::CommandPool m_commandPool;

    };

    class Buffer {
    public:
        Buffer(const Device& device, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
                : m_deviceRef(device), m_size(size) {
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
        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;
        ~Buffer() {
            m_deviceRef.getDevice().destroyBuffer(m_buffer);
            m_deviceRef.getDevice().freeMemory(m_bufferMemory);
        }

        void map(vk::DeviceSize size = VK_WHOLE_SIZE) {
            if (m_mapped != nullptr) return;
            auto res = m_deviceRef.getDevice().mapMemory(m_bufferMemory, 0, size, vk::MemoryMapFlags(), &m_mapped);
            if (res != vk::Result::eSuccess || m_mapped == nullptr) {
                throw std::runtime_error("Failed to map memory");
            }
        }

        void copyTo(void* data, vk::DeviceSize size = VK_WHOLE_SIZE) {
            if (size == VK_WHOLE_SIZE) size = m_size;
            memcpy(m_mapped, data, size);
        }

        void unmap() {
            m_deviceRef.getDevice().unmapMemory(m_bufferMemory);
            m_mapped = nullptr;
        }

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

        void readAll(void* data) {
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

        void writeAll(void* data) {
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

        template<typename T>
        void write(T& data) {
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
        void write(InputIt first, InputIt last) {
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
        void read(T &data) {
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
        void read(InputIt first, InputIt last) {
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

    private:

        const Device& m_deviceRef;
        bool m_isHostBuffer = false;
        bool m_changed = false;
        vk::Buffer m_buffer;
        vk::DeviceMemory m_bufferMemory;
        vk::DeviceSize m_size;
        void* m_mapped = nullptr;
    };

    class DescriptorPool {
    public:
        class Builder {
        public:
            explicit Builder(const Device& device): m_deviceRef(device) {}

            Builder& addPoolSize(vk::DescriptorPoolSize poolSize) {
                m_poolSizes.push_back(poolSize);
                return *this;
            }
            Builder& setFlags(vk::DescriptorPoolCreateFlags flags) { m_poolFlags = flags; return *this; }
            Builder& setMaxSets(uint32_t maxSets) { m_poolMaxSets = maxSets; return *this; }
            Builder& setMaxSetsTimesSizes(uint32_t maxSets) { m_poolMaxSets = maxSets*m_poolSizes.size(); return *this; }
            [[nodiscard]] DescriptorPool build() const {
                return {m_deviceRef, m_poolSizes, m_poolFlags, m_poolMaxSets};
            }

        private:
            const Device& m_deviceRef;
            std::vector<vk::DescriptorPoolSize> m_poolSizes{};
            vk::DescriptorPoolCreateFlags m_poolFlags;
            uint32_t m_poolMaxSets = 1000;
        };

        DescriptorPool(const Device &device, const std::vector<vk::DescriptorPoolSize>& poolSizes, vk::DescriptorPoolCreateFlags poolFlags, uint32_t poolMaxSets)
                : m_deviceRef(device) {
            vk::DescriptorPoolCreateInfo poolInfo({}, poolMaxSets, poolSizes.size(), poolSizes.data());
            m_descriptorPool = m_deviceRef.getDevice().createDescriptorPool(poolInfo);
        }
        ~DescriptorPool() {
            m_deviceRef.getDevice().resetDescriptorPool(m_descriptorPool);
            m_deviceRef.getDevice().destroyDescriptorPool(m_descriptorPool);
        }

        [[nodiscard]] vk::DescriptorPool getDescriptorPool() const { return m_descriptorPool; }

    private:
        const Device& m_deviceRef;

        vk::DescriptorPool m_descriptorPool{};
    };

    class Kernel {
    public:
        Kernel(const Device &device, const std::vector<vk::DescriptorSetLayoutBinding>& layout, std::string  shaderData)
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
        Kernel(Kernel& other) = delete;
        Kernel operator=(Kernel& other) = delete;
        ~Kernel() {
            m_deviceRef.getDevice().destroyDescriptorSetLayout(m_descriptorSetLayout);
            m_deviceRef.getDevice().destroyPipeline(m_pipeline);
            m_deviceRef.getDevice().destroyPipelineLayout(m_pipelineLayout);
            m_deviceRef.getDevice().destroyPipelineCache(m_cache);
            m_deviceRef.getDevice().destroyShaderModule(m_shaderModule);
        }

        void addDescriptorSet(DescriptorPool& pool, const std::vector<vk::DescriptorBufferInfo>& bufferInfos) {
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
        uint32_t findOrAddDescriptorSet(DescriptorPool& pool, const std::vector<vk::DescriptorBufferInfo>& bufferInfos) {
            for (uint32_t i = 0; i < m_bufferInfos.size(); ++i) {
                if (m_bufferInfos[i] == bufferInfos) return i;
            }
            addDescriptorSet(pool, bufferInfos);
            return m_sets.size() - 1;
        }
        void run(vk::CommandBuffer commandBuffer, uint32_t setIdx, uint32_t x, uint32_t y, uint32_t z) const {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, m_sets[setIdx], nullptr);
            commandBuffer.dispatch(x, y, z);
        }

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

    class ComputeHandler {
    public:
        explicit ComputeHandler(const Device &device): m_deviceRef(device) {
            m_fence = m_deviceRef.getDevice().createFence(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
            m_semaphore = m_deviceRef.getDevice().createSemaphore(vk::SemaphoreCreateInfo());
            m_commandBuffer = m_deviceRef.allocateCommandBuffer();
        }
        ~ComputeHandler() {
            m_deviceRef.freeCommandBuffer(m_commandBuffer);
            m_deviceRef.getDevice().destroyFence(m_fence);
            m_deviceRef.getDevice().destroySemaphore(m_semaphore);
        }
        ComputeHandler(const ComputeHandler &) = delete;
        ComputeHandler &operator=(const ComputeHandler &) = delete;

        void computeFrame(const std::function<void(vk::CommandBuffer &)> &function) {
            beginComputeFrame();
            function(m_commandBuffer);
            submitComputeFrame();
        }
        void beginComputeFrame() {
//        auto res = m_deviceRef.getDevice().waitForFences(1, &m_fence, true, UINT64_MAX);
            auto res = m_deviceRef.getDevice().resetFences(1, &m_fence);

            m_commandBuffer.reset(vk::CommandBufferResetFlags());
            m_commandBuffer.begin(vk::CommandBufferBeginInfo());
            m_isComputeFrame = true;
        }
        void submitComputeFrame() {
            m_commandBuffer.end();

            vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &m_commandBuffer);
            m_deviceRef.computeQueue().submit(submitInfo, m_fence);
            auto res = m_deviceRef.getDevice().waitForFences(1, &m_fence, true, UINT64_MAX);

            if (res != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to wait for fences");
            }
            m_isComputeFrame = false;
        }
        static void computeBarrier(vk::CommandBuffer commandBuffer, Buffer& buffer) {
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
        [[nodiscard]] vk::CommandBuffer getCommandBuffer() const {
            if (!m_isComputeFrame) {
                throw std::runtime_error("Not in compute frame");
            }
            return m_commandBuffer;
        }
        [[nodiscard]] bool isComputeFrame() const { return m_isComputeFrame; };

    private:
        bool m_isComputeFrame = false;
        const Device &m_deviceRef;

        vk::Fence m_fence;
        vk::Semaphore m_semaphore;

        vk::CommandBuffer m_commandBuffer{};
    };

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
        VentManager():
                m_descriptorPool(vent::DescriptorPool::Builder(m_device)
                                         .addPoolSize({vk::DescriptorType::eStorageBuffer, 1000})
                                         .setMaxSets(1000)
                                         .build())
        {}
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


    template<std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if<I == sizeof...(Tp), void>::type
    for_each(std::tuple<Tp...> &, FuncT) { }

    template<std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if<I < sizeof...(Tp), void>::type
    for_each(std::tuple<Tp...>& t, FuncT f) {
        f(std::get<I>(t));
        for_each<I + 1, FuncT, Tp...>(t, f);
    }

    enum class ReduceOperation {
        add,
        mul,
        min,
        max,
//        and_,
//        or_,
//        xor_,
    };

    template <typename InputIt>
    std::string getType(InputIt val) {
        // TODO: Add vector support

        if (typeid(decltype(*val)) == typeid(int)) return "int";
        if (typeid(decltype(*val)) == typeid(float)) return "float";
        if (typeid(decltype(*val)) == typeid(uint32_t)) return "uint";
        if (typeid(decltype(*val)) == typeid(int32_t)) return "int";

        throw std::runtime_error("Unsupported type for glsl language");
    }

    template<typename... Tp>
    std::pair<std::string, Buffer&> getUniformBuffer(uint32_t size, std::tuple<Tp...>& args) {
        uint32_t uboSize = sizeof(uint32_t);
        std::stringstream ss;
        ss << R"(
            layout (binding = 0) uniform ParameterUBO {
                uint size;
        )";
        for_each(args, [&uboSize, &ss](auto& p) {
            uboSize += sizeof(p.second);
            ss << vent::getType(&p.second) << " " << p.first << ";\n";
        });

        ss << "};\n";

        char data[uboSize];
        memcpy(data, &size, sizeof(uint32_t));
        uint32_t offset = sizeof(uint32_t);
        for_each(args, [&data, &offset](auto& p) {
            memcpy(data + offset, &p.second, sizeof(p.second));
            offset += sizeof(p.second);
        });

        auto& uniBuffers = VentManager::getInstance().getUniformBuffers();

        if (uniBuffers.find(uboSize) == uniBuffers.end()) {
            uniBuffers.try_emplace(uboSize, VentManager::getInstance().getDevice(), uboSize,
                                   vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible |
                                                                            vk::MemoryPropertyFlagBits::eHostCoherent);
        }
        auto& uniformBuffer = uniBuffers.at(uboSize);
        uniformBuffer.writeAll((void*) data);

        return {ss.str(), uniformBuffer};
    }

    template< class InputIt, class OutputIt, typename... Tp>
    OutputIt isolated_transform( InputIt first1, InputIt last1,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {}, bool isHostBuffer = false){
        auto size = std::distance(first1, last1);

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;
        Buffer inBuffer{VentManager::getInstance().getDevice(),
                        sizeof(*first1) * size,
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, properties};

        inBuffer.write(first1, last1);

        Buffer outBuffer{VentManager::getInstance().getDevice(),
                         sizeof(*d_first) * size,
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, properties};


        std::string strOp = (!operation.empty()) ? operation : getType(d_first) + " transOp(" + getType(first1) + " input1) { return input1; }";
        auto start = strOp.find(' ') + 1;
        strOp.replace(start, strOp.find('(') - start, "transformOp");

        const std::string transformShader = R"(
            #version 460
            )" + uboText + R"(
            layout(std430, binding = 1) readonly buffer InSSBO {
                )" + getType(first1) + R"( inBuffer[ ];
            };

            layout(std430, binding = 2) buffer OutSSBO {
                )" + getType(d_first) + R"( outBuffer[ ];
            };

            )" + strOp + R"(

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= size) return;
                outBuffer[idx] = transformOp(inBuffer[idx]);
            }
        )";

        auto& kernels = VentManager::getInstance().getKernels();
        if (kernels.find(transformShader) == kernels.end()){
            const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                    {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
            };
            kernels.try_emplace(transformShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformShader);
        }

        auto& kernel = kernels.at(transformShader);

        auto idx = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {uniformBuffer.descriptorInfo(), inBuffer.descriptorInfo(), outBuffer.descriptorInfo()});

        VentManager::getInstance().getComputeHandler().computeFrame([&kernel, idx, size](vk::CommandBuffer& commandBuffer){
            kernel.run(commandBuffer, idx, size/256 + 1, 1, 1);
        });

        outBuffer.read(d_first, d_first + size);

        return d_first;
    }

    template< class InputIt1, class InputIt2, class OutputIt, typename... Tp >
    OutputIt isolated_transform( InputIt1 first1, InputIt1 last1, InputIt2 first2,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {},  bool isHostBuffer = false){
        auto size = std::distance(first1, last1);

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;
        Buffer in1Buffer{VentManager::getInstance().getDevice(),
                        sizeof(*first1) * size,
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, properties};
        in1Buffer.write(first1, last1);

        Buffer in2Buffer{VentManager::getInstance().getDevice(),
                         sizeof(*first2) * size,
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, properties};
        in2Buffer.write(first2, first2 + size);


        Buffer outBuffer{VentManager::getInstance().getDevice(),
                         sizeof(*d_first) * size,
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, properties};

        std::string strOp = (!operation.empty()) ? operation : getType(d_first) + " transOp(" +
                getType(first1) + " input1, "+ getType(first2) + "input2) { return input1; }";
        auto start = strOp.find(' ') + 1;
        strOp.replace(start, strOp.find('(') - start, "transformOp");

        const std::string transformShader = R"(
            #version 460
            )" + uboText + R"(
            layout(std430, binding = 1) readonly buffer In1SSBO {
                )" + getType(first1) + R"( in1Buffer[ ];
            };
            layout(std430, binding = 2) readonly buffer In2SSBO {
                )" + getType(first2) + R"( in2Buffer[ ];
            };
            layout(std430, binding = 3) buffer OutSSBO {
                )" + getType(d_first) + R"( outBuffer[ ];
            };

            )" + strOp + R"(

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= size) return;
                outBuffer[idx] = transformOp(in1Buffer[idx], in2Buffer[idx]);
            }
        )";

        auto& kernels = VentManager::getInstance().getKernels();
        if (kernels.find(transformShader) == kernels.end()){
            const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                    {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            };
            kernels.try_emplace(transformShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformShader);
        }

        auto& kernel = kernels.at(transformShader);

        auto idx = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                in2Buffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });
        VentManager::getInstance().getComputeHandler().computeFrame([&kernel, idx, size](vk::CommandBuffer& commandBuffer){
            kernel.run(commandBuffer, idx, size/256 + 1, 1, 1);
        });

        outBuffer.read(d_first, d_first + size);

        return d_first;
    }

    std::string getReduceOp(ReduceOperation reduceOp) {
        switch (reduceOp) {
            case ReduceOperation::add:
                return "subgroupAdd";
            case ReduceOperation::mul:
                return "subgroupMul";
            case ReduceOperation::min:
                return "subgroupMin";
            case ReduceOperation::max:
                return "subgroupMax";
//            case ReduceOperation::and_: reduceFunction = "subgroupAnd"; break;
//            case ReduceOperation::or_: reduceFunction = "subgroupOr"; break;
//            case ReduceOperation::xor_: reduceFunction = "subgroupXor"; break;
        }
        return "";
    }

    std::string createReduceShader(const std::string& ubo, const std::string& type1, const std::string& typeOut, ReduceOperation reduceOp, const std::string& transformOp) {
        std::string reduceFunction = getReduceOp(reduceOp);

        std::string strOp = (!transformOp.empty()) ? transformOp : typeOut + " transOp(" + type1 + " input1) { return input1; }";
        auto start = strOp.find(' ') + 1;
        strOp.replace(start, strOp.find('(') - start, "transformOp");

        return R"(
            #version 450
            #extension GL_KHR_shader_subgroup_arithmetic : enable
            )" + ubo + R"(
            layout(std430, binding = 1) readonly buffer InSSBO {
                )" + type1 + R"( inBuffer[ ];
            };

            layout(std430, binding = 2) buffer OutSSBO {
                )" + typeOut + R"( outBuffer[ ];
            };

            )"+strOp+R"(

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

            const int sumSubGroupSize = 32;
            shared )" + typeOut + R"( sdata[sumSubGroupSize]; // gl_SubgroupSize == 32

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                )" + typeOut + R"( sum = 0;
                if (idx < size) {
                    sum = transformOp(inBuffer[idx]);
                }

                sum = )"+reduceFunction+R"((sum);

                if (gl_SubgroupInvocationID == 0) {
                    sdata[gl_SubgroupID] = sum;
                }

                memoryBarrierShared();
                barrier();

                if (gl_SubgroupID == 0) {
                    sum = (gl_SubgroupInvocationID < gl_NumSubgroups) ? sdata[gl_SubgroupInvocationID] : 0;
                    sum = subgroupAdd(sum);
                }


                if (gl_LocalInvocationID.x == 0) {
                    outBuffer[gl_WorkGroupID.x] = sum;
                }
            }
        )";
    }

    template< class InputIt, class T , typename... Tp>
    T isolated_transform_reduce( InputIt first, InputIt last, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        auto size = std::distance(first, last);

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;
        Buffer inBuffer{VentManager::getInstance().getDevice(),
                        sizeof(*first) * size,
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, properties};

        inBuffer.write(first, last);

        Buffer outBuffer{VentManager::getInstance().getDevice(),
                        sizeof(T) * (size/256 + 1),
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, properties};

        std::string transformRedShader = createReduceShader(uboText, getType(first), getType(&init), reduceOp, transformOp);
        std::string reduceShader = createReduceShader(uboText, getType(&init), getType(&init), reduceOp, "");
        auto& kernels = VentManager::getInstance().getKernels();
        const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        };

        if (kernels.find(transformRedShader) == kernels.end())
            kernels.try_emplace(transformRedShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformRedShader);
        if (kernels.find(reduceShader) == kernels.end())
            kernels.try_emplace(reduceShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, reduceShader);

        auto& initialKernel = kernels.at(transformRedShader);
        auto idx1 = initialKernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                inBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        auto& kernel = kernels.at(reduceShader);
        auto idx2 = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        VentManager::getInstance().getComputeHandler().computeFrame([&kernel, &initialKernel, size, &outBuffer, idx1, idx2](vk::CommandBuffer& commandBuffer){
            initialKernel.run(commandBuffer, idx1, size/256 + 1, 1, 1);

            ComputeHandler::computeBarrier(commandBuffer, outBuffer);

            for (uint32_t n = (size + 255)/256, next = (n + 255)/256; n > 1; n = (n + 255) / 256, next = (n + 255) / 256) {
                kernel.run(commandBuffer, idx2, next, 1, 1);
                ComputeHandler::computeBarrier(commandBuffer, outBuffer);
            }
        });

        T result;
        outBuffer.read(result);

        if (reduceOp == ReduceOperation::add) result += init;
        else if (reduceOp == ReduceOperation::mul) result *= init;
        else if (reduceOp == ReduceOperation::min) result = std::min(result, init);
        else if (reduceOp == ReduceOperation::max) result = std::max(result, init);
        return result;
    }

    std::string createBinaryReduceShader(const std::string& ubo, const std::string& type1, const std::string& type2, const std::string& typeOut, ReduceOperation reduceOp, const std::string& transformOp) {
        std::string reduceFunction = getReduceOp(reduceOp);

        std::string strOp = (!transformOp.empty()) ? transformOp : typeOut + " transOp(" +
                                                               type1 + " x, "+ type2 + " y) { return x*y; }";
        auto start = strOp.find(' ') + 1;
        strOp.replace(start, strOp.find('(') - start, "transformOp");


        return R"(
            #version 450
            #extension GL_KHR_shader_subgroup_arithmetic : enable
            )" + ubo + R"(
            layout(std430, binding = 1) readonly buffer In1SSBO {
                )" + type1 + R"( in1Buffer[ ];
            };

            layout(std430, binding = 2) readonly buffer In2SSBO {
                )" + type2 + R"( in2Buffer[ ];
            };

            layout(std430, binding = 3) buffer OutSSBO {
                )" + typeOut + R"( outBuffer[ ];
            };

            )"+strOp+R"(

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

            const int sumSubGroupSize = 32;
            shared )" + typeOut + R"( sdata[sumSubGroupSize]; // gl_SubgroupSize == 32

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                )" + typeOut + R"( sum = 0;
                if (idx < size) {
                    sum = transformOp(in1Buffer[idx], in2Buffer[idx]);
                }

                sum = )"+reduceFunction+R"((sum);

                if (gl_SubgroupInvocationID == 0) {
                    sdata[gl_SubgroupID] = sum;
                }

                memoryBarrierShared();
                barrier();

                if (gl_SubgroupID == 0) {
                    sum = (gl_SubgroupInvocationID < gl_NumSubgroups) ? sdata[gl_SubgroupInvocationID] : 0;
                    sum = subgroupAdd(sum);
                }


                if (gl_LocalInvocationID.x == 0) {
                    outBuffer[gl_WorkGroupID.x] = sum;
                }
            }
        )";
    }

    template< class InputIt1, class InputIt2, class T, typename... Tp>
    T isolated_transform_reduce( InputIt1 first1, InputIt1 last1, InputIt2 first2, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        auto size = std::distance(first1, last1);

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;
        Buffer in1Buffer{VentManager::getInstance().getDevice(),
                         sizeof(*first1) * size,
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, properties};
        in1Buffer.write(first1, last1);

        Buffer in2Buffer{VentManager::getInstance().getDevice(),
                         sizeof(*first2) * size,
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, properties};
        in2Buffer.write(first2, first2 + size);

        Buffer outBuffer{VentManager::getInstance().getDevice(),
                         sizeof(T) * (size/256 + 1),
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, properties};

        std::string transformRedShader = createBinaryReduceShader(uboText, getType(first1), getType(first2), getType(&init), reduceOp, transformOp);
        std::string reduceShader = createReduceShader(uboText,getType(&init), getType(&init), reduceOp, "");
        auto& kernels = VentManager::getInstance().getKernels();
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        };

        if (kernels.find(reduceShader) == kernels.end())
            kernels.try_emplace(reduceShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, reduceShader);
        if (kernels.find(transformRedShader) == kernels.end()){
            descriptorSetLayoutBinding.emplace_back(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
            kernels.try_emplace(transformRedShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformRedShader);
        }


        auto& initialKernel = kernels.at(transformRedShader);
        auto idx1 = initialKernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                in2Buffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        auto& kernel = kernels.at(reduceShader);
        auto idx2 = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        VentManager::getInstance().getComputeHandler().computeFrame([&kernel, &initialKernel, size, &outBuffer, idx1, idx2](vk::CommandBuffer& commandBuffer){
            initialKernel.run(commandBuffer, idx1, size/256 + 1, 1, 1);

            ComputeHandler::computeBarrier(commandBuffer, outBuffer);

            for (uint32_t n = (size + 255)/256, next = (n + 255)/256; n > 1; n = (n + 255) / 256, next = (n + 255) / 256) {
                kernel.run(commandBuffer, idx2, next, 1, 1);
                ComputeHandler::computeBarrier(commandBuffer, outBuffer);
            }
        });

        T result;
        outBuffer.read(result);

        if (reduceOp == ReduceOperation::add) result += init;
        else if (reduceOp == ReduceOperation::mul) result *= init;
        else if (reduceOp == ReduceOperation::min) result = std::min(result, init);
        else if (reduceOp == ReduceOperation::max) result = std::max(result, init);
        return result;
    }

    template< class InputIt, class T, typename... Tp>
    T isolated_reduce( InputIt first, InputIt last, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        return vent::isolated_transform_reduce(first, last, init, reduceOp, "", args, isHostBuffer);
    }

    void gpu_region(GpuRegionFlags flags, const std::function<void()>& f) {

        if (!(flags & GpuRegionFlags::keepBuffers)) {
            VentManager::getInstance().getBuffers().clear();
            VentManager::getInstance().getUniformBuffers().clear();
        }
        VentManager::getInstance().setGpuRegionFlags(flags);
        VentManager::getInstance().getComputeHandler().beginComputeFrame();
        f();
        VentManager::getInstance().getComputeHandler().submitComputeFrame();

        if (flags & GpuRegionFlags::copyBuffersOut) {
            for (auto& [ptr, buffer] : VentManager::getInstance().getBuffers()) {
                if (buffer.changed()) buffer.readAll((void*)ptr);
            }
        }
        VentManager::getInstance().setGpuRegionFlags(GpuRegionFlags::none);



    }

    template< class InputIt, class OutputIt, typename... Tp>
    OutputIt gpu_transform( InputIt first1, InputIt last1,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        auto size = std::distance(first1, last1);
        auto usages = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto& buffers = VentManager::getInstance().getBuffers();
        bool placed = false;
        if (buffers.find((void*)&(*first1)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first1), VentManager::getInstance().getDevice(), sizeof(*first1) * size, usages, properties);
            placed = true;
        }
        auto& inBuffer = buffers.at((void*)&(*first1));
        if (placed) inBuffer.write(first1, last1);

        if (buffers.find((void*)&(*d_first)) == buffers.end()) {
            buffers.try_emplace((&(*d_first)), VentManager::getInstance().getDevice(),sizeof(*d_first) * size, usages, properties);
        }
        Buffer& outBuffer = buffers.at((void*)&(*d_first));
        outBuffer.setAsChanged();

        std::string strOp = (!operation.empty()) ? operation : getType(d_first) + " transOp(" + getType(first1) + " input1) { return input1; }";
        auto start = strOp.find(' ') + 1;
        strOp.replace(start, strOp.find('(') - start, "transformOp");

        const std::string transformShader = R"(
            #version 460
            )" + uboText + R"(
            layout(std430, binding = 1) readonly buffer InSSBO {
                )" + getType(first1) + R"( inBuffer[ ];
            };

            layout(std430, binding = 2) buffer OutSSBO {
                )" + getType(d_first) + R"( outBuffer[ ];
            };

            )" + strOp + R"(

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= size) return;
                outBuffer[idx] = transformOp(inBuffer[idx]);
            }
        )";

        auto& kernels = VentManager::getInstance().getKernels();
        if (kernels.find(transformShader) == kernels.end()){
            const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                    {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
            };
            kernels.try_emplace(transformShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformShader);
        }

        auto& kernel = kernels.at(transformShader);

        auto idx = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {uniformBuffer.descriptorInfo(), inBuffer.descriptorInfo(), outBuffer.descriptorInfo()});

        auto commandBuffer = VentManager::getInstance().getComputeHandler().getCommandBuffer();
        kernel.run(commandBuffer, idx, size/256 + 1, 1, 1);
        ComputeHandler::computeBarrier(commandBuffer, outBuffer);

        return d_first;
    }

    template< class InputIt1, class InputIt2, class OutputIt, typename... Tp>
    OutputIt gpu_transform( InputIt1 first1, InputIt1 last1, InputIt2 first2,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        auto size = std::distance(first1, last1);
        auto usages = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto& buffers = VentManager::getInstance().getBuffers();
        bool placed = false;
        if (buffers.find((void*)&(*first1)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first1), VentManager::getInstance().getDevice(), sizeof(*first1) * size, usages, properties);
            placed = true;
        }
        auto& in1Buffer = buffers.at((void*)&(*first1));
        if (placed) in1Buffer.write(first1, last1);

        placed = false;
        if (buffers.find((void*)&(*first2)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first2), VentManager::getInstance().getDevice(), sizeof(*first2) * size, usages, properties);
            placed = true;
        }
        auto& in2Buffer = buffers.at((void*)&(*first2));
        if (placed) in2Buffer.write(first2, first2 + size);


        if (buffers.find((void*)&(*d_first)) == buffers.end()) {
            buffers.try_emplace((&(*d_first)), VentManager::getInstance().getDevice(),sizeof(*d_first) * size, usages, properties);
        }
        Buffer& outBuffer = buffers.at((void*)&(*d_first));
        outBuffer.setAsChanged();

        std::string strOp = (!operation.empty()) ? operation : getType(d_first) + " transOp(" +
                                                               getType(first1) + " input1, "+ getType(first2) + "input2) { return input1; }";
        auto start = strOp.find(' ') + 1;
        strOp.replace(start, strOp.find('(') - start, "transformOp");

        const std::string transformShader = R"(
            #version 460
            )" + uboText + R"(
            layout(std430, binding = 1) readonly buffer In1SSBO {
                )" + getType(first1) + R"( in1Buffer[ ];
            };
            layout(std430, binding = 2) readonly buffer In2SSBO {
                )" + getType(first2) + R"( in2Buffer[ ];
            };
            layout(std430, binding = 3) buffer OutSSBO {
                )" + getType(d_first) + R"( outBuffer[ ];
            };

            )" + strOp + R"(

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= size) return;
                outBuffer[idx] = transformOp(in1Buffer[idx], in2Buffer[idx]);
            }
        )";

        auto& kernels = VentManager::getInstance().getKernels();
        if (kernels.find(transformShader) == kernels.end()){
            const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                    {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                    {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            };
            kernels.try_emplace(transformShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformShader);
        }

        auto& kernel = kernels.at(transformShader);

        auto idx = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                in2Buffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        auto commandBuffer = VentManager::getInstance().getComputeHandler().getCommandBuffer();
        kernel.run(commandBuffer, idx, size/256 + 1, 1, 1);
        ComputeHandler::computeBarrier(commandBuffer, outBuffer);

        return d_first;
    }

    template< class InputIt, class T, typename... Tp>
    T gpu_transform_reduce( InputIt first, InputIt last, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        auto size = std::distance(first, last);
        auto usages = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto& buffers = VentManager::getInstance().getBuffers();
        bool placed = false;
        if (buffers.find((void*)&(*first)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first), VentManager::getInstance().getDevice(), sizeof(*first) * size, usages, properties);
            placed = true;
        }
        auto& inBuffer = buffers.at((void*)&(*first));
        if (placed) inBuffer.write(first, last);

        if (buffers.find((void*)sizeof(T)) == buffers.end()) {
            buffers.try_emplace((void*)sizeof(T), VentManager::getInstance().getDevice(), sizeof(T) * (size/256 + 1), usages, properties);
        }
        Buffer& outBuffer = buffers.at((void*)sizeof(T));

        std::string transformRedShader = createReduceShader(uboText, getType(first), getType(&init), reduceOp, transformOp);
        std::string reduceShader = createReduceShader(uboText, getType(&init), getType(&init), reduceOp, "");
        auto& kernels = VentManager::getInstance().getKernels();
        const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        };

        if (kernels.find(transformRedShader) == kernels.end())
            kernels.try_emplace(transformRedShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformRedShader);
        if (kernels.find(reduceShader) == kernels.end())
            kernels.try_emplace(reduceShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, reduceShader);

        auto& initialKernel = kernels.at(transformRedShader);
        auto idx1 = initialKernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                inBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        auto& kernel = kernels.at(reduceShader);
        auto idx2 = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        auto commandBuffer = VentManager::getInstance().getComputeHandler().getCommandBuffer();
        initialKernel.run(commandBuffer, idx1, size/256 + 1, 1, 1);
        ComputeHandler::computeBarrier(commandBuffer, outBuffer);

        for (uint32_t n = (size + 255)/256, next = (n + 255)/256; n > 1; n = (n + 255) / 256, next = (n + 255) / 256) {
            kernel.run(commandBuffer, idx2, next, 1, 1);
            ComputeHandler::computeBarrier(commandBuffer, outBuffer);
        }

        VentManager::getInstance().getComputeHandler().submitComputeFrame();

        T result;
        outBuffer.read(result);
        if (reduceOp == ReduceOperation::add) result += init;
        else if (reduceOp == ReduceOperation::mul) result *= init;
        else if (reduceOp == ReduceOperation::min) result = std::min(result, init);
        else if (reduceOp == ReduceOperation::max) result = std::max(result, init);

        VentManager::getInstance().getComputeHandler().beginComputeFrame();
        return result;
    }

    template< class InputIt1, class InputIt2, class T, typename... Tp>
    T gpu_transform_reduce( InputIt1 first1, InputIt1 last1, InputIt2 first2, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        auto size = std::distance(first1, last1);
        auto usages = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;

        auto [uboText, uniformBuffer] = getUniformBuffer(size, args);

        auto& buffers = VentManager::getInstance().getBuffers();
        bool placed = false;
        if (buffers.find((void*)&(*first1)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first1), VentManager::getInstance().getDevice(), sizeof(*first1) * size, usages, properties);
            placed = true;
        }
        auto& in1Buffer = buffers.at((void*)&(*first1));
        if (placed) in1Buffer.write(first1, last1);

        placed = false;
        if (buffers.find((void*)&(*first2)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first2), VentManager::getInstance().getDevice(), sizeof(*first2) * size, usages, properties);
            placed = true;
        }
        auto& in2Buffer = buffers.at((void*)&(*first2));
        if (placed) in2Buffer.write(first2, first2 + size);

        if (buffers.find((void*)sizeof(T)) == buffers.end()) {
            buffers.try_emplace((void*)sizeof(T), VentManager::getInstance().getDevice(), sizeof(T) * (size/256 + 1), usages, properties);
        }
        Buffer& outBuffer = buffers.at((void*)sizeof(T));

        std::string transformRedShader = createBinaryReduceShader(uboText, getType(first1), getType(first2), getType(&init), reduceOp, transformOp);
        std::string reduceShader = createReduceShader(uboText, getType(&init), getType(&init), reduceOp, "");

        auto& kernels = VentManager::getInstance().getKernels();
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        };

        if (kernels.find(reduceShader) == kernels.end())
            kernels.try_emplace(reduceShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, reduceShader);
        if (kernels.find(transformRedShader) == kernels.end()){
            descriptorSetLayoutBinding.emplace_back(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
            kernels.try_emplace(transformRedShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding, transformRedShader);
        }


        auto& initialKernel = kernels.at(transformRedShader);
        auto idx1 = initialKernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                in2Buffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        auto& kernel = kernels.at(reduceShader);
        auto idx2 = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        auto commandBuffer = VentManager::getInstance().getComputeHandler().getCommandBuffer();
        initialKernel.run(commandBuffer, idx1, size/256 + 1, 1, 1);
        ComputeHandler::computeBarrier(commandBuffer, outBuffer);

        for (uint32_t n = (size + 255)/256, next = (n + 255)/256; n > 1; n = (n + 255) / 256, next = (n + 255) / 256) {
            kernel.run(commandBuffer, idx2, next, 1, 1);
            ComputeHandler::computeBarrier(commandBuffer, outBuffer);
        }

        VentManager::getInstance().getComputeHandler().submitComputeFrame();

        T result;
        outBuffer.read(result);
        if (reduceOp == ReduceOperation::add) result += init;
        else if (reduceOp == ReduceOperation::mul) result *= init;
        else if (reduceOp == ReduceOperation::min) result = std::min(result, init);
        else if (reduceOp == ReduceOperation::max) result = std::max(result, init);

        VentManager::getInstance().getComputeHandler().beginComputeFrame();
        return result;
    }

    template< class InputIt, class T, typename... Tp>
    T gpu_reduce( InputIt first, InputIt last, T init,
              ReduceOperation reduceOp = ReduceOperation::add, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        return vent::gpu_transform_reduce(first, last, init, reduceOp, "", args, isHostBuffer);
    }

    /* @brief Transforms the elements in the range [first1, last1) using the
     *        operation provided and writes the result to the range starting at
     *        d_first.
     *
     *  @param first1  Input iterator to the beginning of the range.
     *  @param last1   Input iterator to the end of the range.
     *  @param d_first Output iterator to the beginning of the destination range.
     *  @param operation The operation to be applied to the elements.
     *                 The operation should be string that represents a function in
     *                 glsl that takes a single argument of the same
     *                 type as the elements in the input range and returns a value of
     *                 the same type of the output range.
     *  @param args     Additional arguments to be passed to the operation.
     *  @param isHostBuffer If true, the buffer will be host visible and coherent.
     *  @return Output iterator to the beginning of the destination range.
     */
    template< class InputIt, class OutputIt, typename... Tp>
    OutputIt transform( InputIt first1, InputIt last1,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform(first1, last1, d_first, operation, args, isHostBuffer);
        }
        return isolated_transform(first1, last1, d_first, operation, args, isHostBuffer);
    }

    /* @brief Transforms the elements in the range [first1, last1) and [first2, first2 + (last1 - first1)) using the
     *        operation provided and writes the result to the range starting at d_first.
     *
     *  @param first1  Input iterator to the beginning of the first range.
     *  @param last1   Input iterator to the end of the first range.
     *  @param first2  Input iterator to the beginning of the second range.
     *  @param d_first Output iterator to the beginning of the destination range.
     *  @param operation The operation to be applied to the elements.
     *                 The operation should be string that represents a function in
     *                 glsl that takes two arguments of the same
     *                 type as the elements in the input ranges and returns a value of
     *                 the same type of the output range.
     *  @param args     Additional arguments to be passed to the operation.
     *  @param isHostBuffer If true, the buffer will be host visible and coherent.
     *  @return Output iterator to the beginning of the destination range.
     */
    template< class InputIt1, class InputIt2, class OutputIt, typename... Tp>
    OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform(first1, last1, first2, d_first, operation, args, isHostBuffer);
        }
        return isolated_transform(first1, last1, first2, d_first, operation, args, isHostBuffer);
    }

    /* @brief Transforms the elements in the range [first, last) using the
     *        operation provided and applies a reduction operation, returning the result.
     *
     *  @param first  Input iterator to the beginning of the range.
     *  @param last   Input iterator to the end of the range.
     *  @param init   Initial value of the reduction operation.
     *  @param reduceOp The reduction operation to be applied to the elements.
     *  @param operation The operation to be applied to the elements.
     *                 The operation should be string that represents a function in
     *                 glsl that takes a single argument of the same
     *                 type as the elements in the input range and returns a value of
     *                 the same type of the output range.
     *  @param args     Additional arguments to be passed to the operation.
     *  @param isHostBuffer If true, the buffer will be host visible and coherent.
     *  @return The result of the reduction operation.
     */
    template< class InputIt, class T, typename... Tp>
    T transform_reduce( InputIt first, InputIt last, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform_reduce(first, last, init, reduceOp, transformOp, args, isHostBuffer);
        }
        return isolated_transform_reduce(first, last, init, reduceOp, transformOp, args, isHostBuffer);
    }

    /* @brief Transforms the elements in the range [first1, last1) and [first2, first2 + (last1 - first1)) using the
     *        operation provided and applies a reduction operation, returning the result.
     *
     * If no operation is provided, the function will perform a inner product operation.
     *
     *  @param first1  Input iterator to the beginning of the first range.
     *  @param last1   Input iterator to the end of the first range.
     *  @param first2  Input iterator to the beginning of the second range.
     *  @param init    Initial value of the reduction operation.
     *  @param reduceOp The reduction operation to be applied to the elements.
     *  @param operation The operation to be applied to the elements.
     *                 The operation should be string that represents a function in
     *                 glsl that takes two arguments of the same
     *                 type as the elements in the input ranges and returns a value of
     *                 the same type of the output range.
     *  @param args     Additional arguments to be passed to the operation.
     *  @param isHostBuffer If true, the buffer will be host visible and coherent.
     *  @return The result of the reduction operation.
     */
    template< class InputIt1, class InputIt2, class T, typename... Tp>
    T transform_reduce( InputIt1 first1, InputIt1 last1, InputIt2 first2, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform_reduce(first1, last1, first2, init, reduceOp, transformOp, args, isHostBuffer);
        }
        return isolated_transform_reduce(first1, last1, first2, init, reduceOp, transformOp, args, isHostBuffer);
    }

    /* @brief Applies a reduction operation to the elements in the range [first, last).
     *
     *  @param first  Input iterator to the beginning of the range.
     *  @param last   Input iterator to the end of the range.
     *  @param init   Initial value of the reduction operation.
     *  @param reduceOp The reduction operation to be applied to the elements.
     *  @param args     Additional arguments to be passed to the operation.
     *  @param isHostBuffer If true, the buffer will be host visible and coherent.
     *  @return The result of the reduction operation.
     */
    template< class InputIt, class T, typename... Tp>
    T reduce( InputIt first, InputIt last, T init,
              ReduceOperation reduceOp = ReduceOperation::add,  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_reduce(first, last, init, reduceOp, {}, isHostBuffer);
        }
        return isolated_reduce(first, last, init, reduceOp, {}, isHostBuffer);
    }



    template< class InputIt1, class InputIt2, class OutputIt>
    OutputIt jacobi( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, uint32_t maxIters, double epsilon, bool isHostBuffer) {
        auto matrixSize = std::distance(first1, last1);
        auto vecSize = (uint32_t) std::sqrt(matrixSize);
        auto usages = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                      vk::BufferUsageFlagBits::eTransferSrc;
        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible |
                                           vk::MemoryPropertyFlagBits::eHostCoherent
                                         : vk::MemoryPropertyFlagBits::eDeviceLocal;

        uint32_t uboSize = sizeof(uint32_t);
        char data[uboSize];
        memcpy(data, &vecSize, sizeof(uint32_t));
        auto& uniBuffers = VentManager::getInstance().getUniformBuffers();

        if (uniBuffers.find(uboSize) == uniBuffers.end()) {
            uniBuffers.try_emplace(uboSize, VentManager::getInstance().getDevice(), uboSize,
                                   vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible |
                                                                            vk::MemoryPropertyFlagBits::eHostCoherent);
        }
        auto& uniformBuffer = uniBuffers.at(uboSize);
        uniformBuffer.writeAll((void*) data);

        auto& buffers = VentManager::getInstance().getBuffers();
        bool placed = false;
        if (buffers.find((void*)&(*first1)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first1), VentManager::getInstance().getDevice(), sizeof(*first1) * matrixSize, usages, properties);
            placed = true;
        }
        auto& in1Buffer = buffers.at((void*)&(*first1));
        if (placed) in1Buffer.write(first1, last1);

        placed = false;
        if (buffers.find((void*)&(*first2)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first2), VentManager::getInstance().getDevice(), sizeof(*first2) * vecSize, usages, properties);
            placed = true;
        }
        auto& in2Buffer = buffers.at((void*)&(*first2));
        if (placed) in2Buffer.write(first2, first2 + vecSize);

        Buffer tempBuffer{VentManager::getInstance().getDevice(), sizeof(*d_first) * vecSize, usages, properties};
        tempBuffer.setAsChanged();

        if (buffers.find((void*)&(*d_first)) == buffers.end()) {
            buffers.try_emplace((&(*d_first)), VentManager::getInstance().getDevice(),sizeof(*d_first) * vecSize, usages, properties);
        }
        Buffer& outBuffer = buffers.at((void*)&(*d_first));
        outBuffer.setAsChanged();

        const std::string linsolveShader = R"(
            #version 460
            layout (binding = 0) uniform ParameterUBO {
                uint size;
            };
            layout(std430, binding = 1) readonly buffer In1SSBO {
                )" + getType(d_first) + R"( matrix[ ];
            };
            layout(std430, binding = 2) readonly buffer In2SSBO {
                )" + getType(d_first) + R"( b[ ];
            };
            layout(std430, binding = 3) buffer OutSSBO {
                )" + getType(d_first) + R"( outBuffer[ ];
            };
            layout(std430, binding = 4) buffer TempSSBO {
                )" + getType(d_first) + R"( tempBuffer[ ];
            };

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= size) return;
                )" + getType(d_first) + R"( res = b[idx];
                for (uint i = 0; i < size; i++) {
                     if (idx != i) res -= matrix[idx*size + i]*tempBuffer[i];
                }
                outBuffer[idx] = res/matrix[idx*size + idx];
            }
        )";

        auto &kernels = VentManager::getInstance().getKernels();
        const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        if (kernels.find(linsolveShader) == kernels.end()) {
            kernels.try_emplace(linsolveShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding,
                                linsolveShader);
        }

        auto &kernel = kernels.at(linsolveShader);
        auto idx = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                in2Buffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
                tempBuffer.descriptorInfo(),
        });
        auto idx2 = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                in2Buffer.descriptorInfo(),
                tempBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        std::string errorShader = R"(
            #version 460
            layout (binding = 0) uniform ParameterUBO {
                uint size;
            };
            layout(std430, binding = 1) readonly buffer In1SSBO {
                )" + getType(d_first) + R"( matrix[ ];
            };
            layout(std430, binding = 2) readonly buffer In2SSBO {
                )" + getType(d_first) + R"( b[ ];
            };
            layout(std430, binding = 3) buffer OutSSBO {
                )" + getType(d_first) + R"( outBuffer[ ];
            };
            layout(std430, binding = 4) buffer TempSSBO {
                )" + getType(d_first) + R"( tempBuffer[ ];
            };

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= size) return;
                )" + getType(d_first) + R"( res = 0;
                for (uint i = 0; i < size; i++) {
                     if (i != idx) res += matrix[idx*size + i]*outBuffer[i];
                }
                tempBuffer[idx] = (b[idx] - (res + matrix[idx * size + idx] * outBuffer[idx]))/b[idx];
            }
        )";
        if (kernels.find(errorShader) == kernels.end()) {
            kernels.try_emplace(errorShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding,
                                errorShader);
        }

        auto &errorKernel = kernels.at(errorShader);
        auto errorKernelIdx = errorKernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                in2Buffer.descriptorInfo(),
                tempBuffer.descriptorInfo(),
                outBuffer.descriptorInfo(),
        });

        gpu_region(GpuRegionFlags::keepBuffers | GpuRegionFlags::copyBuffersOut, [&]() {
            auto commandBuffer = VentManager::getInstance().getComputeHandler().getCommandBuffer();

            uint32_t iter = 0;
            double errorSq = 100;
            uint32_t gpuIters = 10; // has to be even
            do {

                for (uint32_t i = 0; i < gpuIters; i++) {
                    kernel.run(commandBuffer, (i % 2 == 0) ? idx : idx2, vecSize / 256 + 1, 1, 1);
                    if (i%2 == 0) ComputeHandler::computeBarrier(commandBuffer, outBuffer);
                    else ComputeHandler::computeBarrier(commandBuffer, tempBuffer);
                }

                errorKernel.run(commandBuffer, errorKernelIdx, vecSize / 256 + 1, 1, 1);
                ComputeHandler::computeBarrier(commandBuffer, outBuffer);

                errorSq = vent::reduce(d_first, d_first + vecSize, 0.0f, ReduceOperation::max);

                iter += gpuIters;
            } while (iter < maxIters && errorSq > epsilon);

            outBuffer.swap(tempBuffer);
        });

        return d_first;
    }

    template< class InputIt1, class InputIt2, class OutputIt>
    OutputIt cg( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, uint32_t maxIters, double epsilon, bool isHostBuffer) {
        auto matrixSize = std::distance(first1, last1);
        auto vecSize = (uint32_t) std::sqrt(matrixSize);
        auto usages = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                      vk::BufferUsageFlagBits::eTransferSrc;
        auto properties = (isHostBuffer) ? vk::MemoryPropertyFlagBits::eHostVisible |
                                           vk::MemoryPropertyFlagBits::eHostCoherent
                                         : vk::MemoryPropertyFlagBits::eDeviceLocal;

        uint32_t uboSize = sizeof(uint32_t);
        char data[uboSize];
        memcpy(data, &vecSize, sizeof(uint32_t));
        auto& uniBuffers = VentManager::getInstance().getUniformBuffers();

        if (uniBuffers.find(uboSize) == uniBuffers.end()) {
            uniBuffers.try_emplace(uboSize, VentManager::getInstance().getDevice(), uboSize,
                                   vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible |
                                                                            vk::MemoryPropertyFlagBits::eHostCoherent);
        }
        auto& uniformBuffer = uniBuffers.at(uboSize);
        uniformBuffer.writeAll((void*) data);

        auto& buffers = VentManager::getInstance().getBuffers();
        bool placed = false;
        if (buffers.find((void*)&(*first1)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first1), VentManager::getInstance().getDevice(), sizeof(*first1) * matrixSize, usages, properties);
            placed = true;
        }
        auto& in1Buffer = buffers.at((void*)&(*first1));
        if (placed) in1Buffer.write(first1, last1);

        placed = false;
        if (buffers.find((void*)&(*first2)) == buffers.end()) {
            buffers.try_emplace((void*) &(*first2), VentManager::getInstance().getDevice(), sizeof(*first2) * vecSize, usages, properties);
            placed = true;
        }
        auto& in2Buffer = buffers.at((void*)&(*first2));
        if (placed) in2Buffer.write(first2, first2 + vecSize);

        if (buffers.find((void*)&(*d_first)) == buffers.end()) {
            buffers.try_emplace((&(*d_first)), VentManager::getInstance().getDevice(),sizeof(*d_first) * vecSize, usages, properties);
        }
        Buffer& outBuffer = buffers.at((void*)&(*d_first));
        outBuffer.setAsChanged();

        std::vector<float> s(vecSize, 0);
        std::vector<float> r(vecSize, 0);
        std::vector<float> as(vecSize, 0);

        buffers.try_emplace((&(*s.begin())), VentManager::getInstance().getDevice(),sizeof(*s.begin()) * vecSize, usages, properties);
        Buffer& sBuffer = buffers.at((void*)&(*s.begin()));
        buffers.try_emplace((&(*as.begin())), VentManager::getInstance().getDevice(),sizeof(*as.begin()) * vecSize, usages, properties);
        Buffer& asBuffer = buffers.at((void*)&(*as.begin()));
        buffers.try_emplace((&(*r.begin())), VentManager::getInstance().getDevice(),sizeof(*r.begin()) * vecSize, usages, properties);
        Buffer& rBuffer = buffers.at((void*)&(*r.begin()));

        const std::string matVecShader = R"(
            #version 460
            layout (binding = 0) uniform ParameterUBO {
                uint size;
            };
            layout(std430, binding = 1) readonly buffer In1SSBO {
                )" + getType(d_first) + R"( matrix[ ];
            };
            layout(std430, binding = 2) readonly buffer In2SSBO {
                )" + getType(d_first) + R"( vecIn[ ];
            };
            layout(std430, binding = 3) buffer OutSSBO {
                )" + getType(d_first) + R"( result[ ];
            };

            layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= size) return;
                )" + getType(d_first) + R"( res = 0;
                for (uint i = 0; i < size; i++) {
                     res += matrix[idx*size + i]*vecIn[i];
                }
                result[idx] = res;
            }
        )";

        auto &kernels = VentManager::getInstance().getKernels();
        const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
                {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        if (kernels.find(matVecShader) == kernels.end()) {
            kernels.try_emplace(matVecShader, VentManager::getInstance().getDevice(), descriptorSetLayoutBinding,
                                matVecShader);
        }

        auto &kernel = kernels.at(matVecShader);
        auto idx = kernel.findOrAddDescriptorSet(VentManager::getInstance().getDescriptorPool(), {
                uniformBuffer.descriptorInfo(),
                in1Buffer.descriptorInfo(),
                sBuffer.descriptorInfo(),
                asBuffer.descriptorInfo(),
        });

        gpu_region(GpuRegionFlags::keepBuffers | GpuRegionFlags::copyBuffersOut, [&]() {
            auto commandBuffer = VentManager::getInstance().getComputeHandler().getCommandBuffer();
            // s = r = b
            vent::transform(first2, first2 + vecSize, s.begin(), "float func(float x) {return x; }");
            vent::transform(first2, first2 + vecSize, r.begin(), "float func(float x) {return x; }");

            auto initialDot = vent::transform_reduce(r.begin(), r.end(), r.begin(), 0.0f);

            for (uint32_t iter = 0; iter < maxIters; iter++){
                // as = A*s
                kernel.run(commandBuffer, idx, vecSize / 256 + 1, 1, 1);
                ComputeHandler::computeBarrier(commandBuffer, asBuffer);

                auto sas = vent::transform_reduce(s.begin(), s.end(), as.begin(), 0.0f);
                auto rr = (iter == 0) ? initialDot : vent::transform_reduce(r.begin(), r.end(), r.begin(), 0.0f);

                if (rr < epsilon*initialDot) {
                    break;
                }

                auto alpha = rr / sas;

                // x = x + alpha*s
                vent::transform(d_first, d_first + vecSize, s.begin(), d_first, "float func(float x, float s) {return x + alpha*s; }", std::make_tuple(std::make_pair("alpha", alpha)));

                // r = r - alpha*as
                vent::transform(r.begin(), r.end(), as.begin(), r.begin(), "float func(float r, float as) {return r - alpha*as; }", std::make_tuple(std::make_pair("alpha", alpha)));

                auto prevRr = rr;
                rr = vent::transform_reduce(r.begin(), r.end(), r.begin(), 0.0f);
                auto beta = rr / prevRr;

                // s = r + beta*s
                vent::transform(s.begin(), s.end(), r.begin(), s.begin(), "float func(float s, float r) {return r + beta*s; }", std::make_tuple(std::make_pair("beta", beta)));
            }

        });

        outBuffer.read(d_first, d_first + vecSize);

        return d_first;
    }

    /* @brief Solves the linear system of equations Ax = b.
     *
     * If the matrix is positive definite, the conjugate gradient method is used.
     * Otherwise, the Jacobi method is used.
     *
     *  @param first1  Input iterator to the beginning of the matrix.
     *  @param last1   Input iterator to the end of the matrix.
     *  @param first2  Input iterator to the beginning of the vector.
     *  @param d_first Output iterator to the beginning of the destination vector.
     *  @param maxIters Maximum number of iterations.
     *  @param epsilon  Error tolerance.
     *  @param isPositiveDefinite If true, the conjugate gradient method will be used.
     *  @param isHostBuffer If true, the buffer will be host visible and coherent.
     *  @return Output iterator to the beginning of the destination vector.
     */
    template< class InputIt1, class InputIt2, class OutputIt>
    OutputIt linsolve( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, uint32_t maxIters, double epsilon, bool isPositiveDefinite = false, bool isHostBuffer = false) {
        if (isPositiveDefinite) {
            return cg(first1, last1, first2, d_first, maxIters, epsilon, isHostBuffer);
        }
        return jacobi(first1, last1, first2, d_first, maxIters, epsilon,  isHostBuffer);
    }

}



#endif //VULKANCOMPUTEPLAYGROUND_VENT_H
