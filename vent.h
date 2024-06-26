//
// Created by luc on 11/06/24.
//

#ifndef VULKANCOMPUTEPLAYGROUND_VENT_H
#define VULKANCOMPUTEPLAYGROUND_VENT_H

#include <vector>
#include "Buffer.h"
#include "VentManager.h"
#include "Kernel.h"

namespace vent {

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
        std::cout << "heyy\n";
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

    template< class InputIt, class OutputIt, typename... Tp>
    OutputIt transform( InputIt first1, InputIt last1,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform(first1, last1, d_first, operation, args, isHostBuffer);
        }
        return isolated_transform(first1, last1, d_first, operation, args, isHostBuffer);
    }

    template< class InputIt1, class InputIt2, class OutputIt, typename... Tp>
    OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2,
                        OutputIt d_first, const std::string& operation, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform(first1, last1, first2, d_first, operation, args, isHostBuffer);
        }
        return isolated_transform(first1, last1, first2, d_first, operation, args, isHostBuffer);
    }

    template< class InputIt, class T, typename... Tp>
    T transform_reduce( InputIt first, InputIt last, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform_reduce(first, last, init, reduceOp, transformOp, args, isHostBuffer);
        }
        return isolated_transform_reduce(first, last, init, reduceOp, transformOp, args, isHostBuffer);
    }

    template< class InputIt1, class InputIt2, class T, typename... Tp>
    T transform_reduce( InputIt1 first1, InputIt1 last1, InputIt2 first2, T init,
                        ReduceOperation reduceOp = ReduceOperation::add, const std::string& transformOp = "", std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_transform_reduce(first1, last1, first2, init, reduceOp, transformOp, args, isHostBuffer);
        }
        return isolated_transform_reduce(first1, last1, first2, init, reduceOp, transformOp, args, isHostBuffer);
    }

    template< class InputIt, class T, typename... Tp>
    T reduce( InputIt first, InputIt last, T init,
              ReduceOperation reduceOp = ReduceOperation::add, std::tuple<Tp...> args = {},  bool isHostBuffer = false) {
        if (VentManager::getInstance().getComputeHandler().isComputeFrame()) {
            return gpu_reduce(first, last, init, reduceOp, args, isHostBuffer);
        }
        return isolated_reduce(first, last, init, reduceOp, args, isHostBuffer);
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


    template< class InputIt1, class InputIt2, class OutputIt>
    OutputIt linsolve( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, uint32_t maxIters, double epsilon, bool isPositiveDefinite = false, bool isHostBuffer = false) {
        if (isPositiveDefinite) {
            return cg(first1, last1, first2, d_first, maxIters, epsilon, isHostBuffer);
        }
        return jacobi(first1, last1, first2, d_first, maxIters, epsilon,  isHostBuffer);
    }

}



#endif //VULKANCOMPUTEPLAYGROUND_VENT_H
