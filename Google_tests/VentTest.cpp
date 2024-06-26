#include <complex>
#include <numeric>
#include <random>
#include <array>
#include <iostream>
#include <list>
#include <string>
#include <tuple>
#include <vector>
#include "gtest/gtest.h"
#include "../vent.h"

template<typename T>
T error(T a, T b) {
    return std::abs(a - b)/b;
}

inline float randomFloat(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis_float;
    return dis_float(gen);
}

inline float randomFloat(float min, float max) {
    return min + randomFloat() * (max - min);
}

TEST(VentTest, UnaryTransform) {
    const int n = 10;
    std::vector<int> d1(n);
    for (int i = 0; i < n; ++i) {
        d1[i] = i;
    }
    std::vector<float> outData(n);
    int multiplier = 3;
    vent::transform(d1.begin(), d1.end(), outData.begin(), "float res(int val) {return float(multiplier*val*val);}",
                    std::make_tuple(std::make_pair("multiplier", multiplier)));

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(outData[i], float(multiplier*d1[i]*d1[i]));
    }

    const uint32_t numElements = 10'000'000;
    std::vector<uint32_t> inData(numElements);
    for (int i = 0; i < numElements; ++i) {
        inData[i] = i;
    }
    std::vector<float> outData2(numElements);


    auto start = std::chrono::high_resolution_clock::now();
    vent::transform(inData.begin(), inData.end(), outData2.begin(), "float res(uint val) {return cos(2.0f * 3.14f * float(val)/10000000.0);}");
    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Gpu Time: " << gpuTime << "s" << std::endl;

    std::vector<uint32_t> v(numElements);
    for (int i = 0; i < numElements; ++i) {
        v[i] = i;
    }
    std::vector<float> v2(numElements);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numElements; ++i)
    {
        v2[i] = std::cos(2.0f * 3.14f * float(v[i])/float(numElements));
    }
    end = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Cpu Time: " << cpuTime << "s" << std::endl;
    std::cout << v2[numElements - 1] << std::endl;

    std::cout << "speedup: " << cpuTime/gpuTime << "\n";
}

TEST(VentTest, BinaryTransform) {
    uint32_t numElements = 20;
    std::vector<uint32_t> inData1(numElements);
    for (int i = 0; i < numElements; ++i) {
        inData1[i] = i;
    }
    std::vector<uint32_t> inData2(numElements);
    for (int i = 0; i < numElements; ++i) {
        inData2[i] = 2*i;
    }
    std::vector<uint32_t> outData(numElements);
    vent::transform(inData1.begin(), inData1.end(), inData2.begin(), outData.begin(), "uint f(uint a, uint b) {return b - a;}");
    for (int i = 0; i < numElements; i++) {
        ASSERT_EQ(outData[i], inData1[i]);
    }

    vent::transform(inData1.begin(), inData1.end(), inData2.begin(), outData.begin(), "uint f(uint a, uint b) {return b + a;}");
    for (int i = 0; i < numElements; i++) {
        ASSERT_EQ(outData[i], 3*inData1[i]);
    }

    numElements = 20'000'000;
    inData1.resize(numElements);
    for (int i = 0; i < numElements; ++i) {
        inData1[i] = i;
    }
    inData2.resize(numElements);
    for (int i = 0; i < numElements; ++i) {
        inData2[i] = 2*i;
    }
    outData.resize(numElements);

    auto start = std::chrono::high_resolution_clock::now();
    vent::transform(inData1.begin(), inData1.end(), inData2.begin(), outData.begin(), "uint f(uint a, uint b) {return b - a;}");
    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "first Gpu Time: " << gpuTime << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vent::transform(inData1.begin(), inData1.end(), inData2.begin(), outData.begin(), "uint f(uint a, uint b) {return b + a;}");
    end = std::chrono::high_resolution_clock::now();
    gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Gpu Time: " << gpuTime << "s" << std::endl;

    std::vector<uint32_t> v2(numElements);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numElements; ++i)
    {
        v2[i] = inData1[i] + inData2[i];
    }
    end = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Cpu Time: " << cpuTime << "s" << std::endl;
    std::cout << v2[numElements - 1] << std::endl;

    std::cout << "speedup: " << cpuTime/gpuTime << "\n";
}

TEST(VentTest, UnaryTransformReduce) {
    const uint32_t numElements = 10'000'000;
    std::vector<float > inData(numElements);
    for (int i = 0; i < numElements; ++i) {
        inData[i] = 1.0f/float(numElements);
    }

    float multiplier = 4.0f;
    auto other = vent::transform_reduce(inData.begin(), inData.end(), 0.0f,
                                        vent::ReduceOperation::add, "float f(float a) {return multiplier*a;}",
                                        std::make_tuple(std::make_pair("multiplier", multiplier)));

    std::cout << other << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = vent::transform_reduce(inData.begin(), inData.end(), 0.0f,
                                      vent::ReduceOperation::add,"float f(float a) {return multiplier*a;}",
                                      std::make_tuple(std::make_pair("multiplier", multiplier)));

    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "first Gpu Time: " << gpuTime << "s" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto sum2 = std::transform_reduce(inData.begin(), inData.end(), 0.0f, std::plus<>(), [multiplier](float f) {return multiplier*f;});
    end = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Cpu Time: " << cpuTime << "s" << std::endl;
    std::cout << "Sum: " << sum2 << std::endl;

    std::cout << "speedup: " << cpuTime/gpuTime << "\n";

    ASSERT_LT(std::abs(sum - sum2), 0.2);
}

TEST(VentTest, BinaryTransformReduce){
    const uint32_t numElements = 10'000'000;
    std::vector<float > in1Data(numElements);
    std::vector<float > in2Data(numElements);
    for (int i = 0; i < numElements; ++i) {
        in1Data[i] = float(i + 1)/float(numElements);
        in2Data[i] = float(numElements - i + 1)/float(numElements);
    }

    auto other = vent::transform_reduce(in1Data.begin(), in1Data.end(), in2Data.begin(), 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = vent::transform_reduce(in1Data.begin(), in1Data.end(), in2Data.begin(), 0.0f);

    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "first Gpu Time: " << gpuTime << "s" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto sum2 = std::transform_reduce(in1Data.begin(), in1Data.end(), in2Data.begin(), 0.0f);
    end = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Cpu Time: " << cpuTime << "s" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    ASSERT_LT(std::abs(sum - sum2)/sum2, 0.01);

    std::cout << "speedup: " << cpuTime/gpuTime << "\n";
}

TEST(VentTest, Reduce){
    const uint32_t numElements = 10;
    std::vector<float > inData(numElements);
    for (int i = 0; i < numElements; ++i) {
        inData[i] = 1.0f/float(i+2);
    }

    auto other = vent::reduce(inData.begin(), inData.end(), 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = vent::reduce(inData.begin(), inData.end(), 0.0f, vent::ReduceOperation::max);
    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "first Gpu Time: " << gpuTime << "s" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto sum2 = std::reduce(inData.begin(), inData.end(), 0.0f, [](float a, float b) {return std::max(a, b);});
    end = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Cpu Time: " << cpuTime << "s" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    ASSERT_EQ(sum, sum2);

    std::cout << "speedup: " << cpuTime/gpuTime << "\n";
}

TEST(VentTest, GpuRegionTransform) {
    const uint32_t n = 100'000;
    std::vector<uint32_t> d1(n);
    for (int i = 0; i < n; ++i) {
        d1[i] = i;
    }
    std::vector<float> outData(n);
    vent::gpu_region(vent::GpuRegionFlags::copyBuffersOut, [&]() {
        vent::transform(d1.begin(), d1.end(), outData.begin(), "float res(uint val) {return float(val*val);}");
        vent::transform(outData.begin(), outData.end(), d1.begin(), outData.begin(), "float res(float a, uint b) {return float(b) + a;}");
        vent::transform(outData.begin(), outData.end(), outData.begin(), "float res(float a) {return a + 1;}");
    });

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(outData[i], float(d1[i]*d1[i]) + float(d1[i]) + float(1));
    }
}

TEST(VentTest, GpuRegionGeneral) {
    const uint32_t n = 1000;
    float multiplier = 1/10.0f;
    std::vector<uint32_t> d1(n);
    for (int i = 0; i < n; ++i) {
        d1[i] = i;
    }
    float sum = 0;
    std::vector<float> outData(n);
    vent::gpu_region(vent::GpuRegionFlags::copyBuffersOut, [&]() {
        vent::transform(d1.begin(), d1.end(), outData.begin(), "float res(uint val) {return float(val*val);}");
        vent::transform(outData.begin(), outData.end(), d1.begin(), outData.begin(), "float res(float a, uint b) {return multiplier*(float(b) + a);}",
                        std::make_tuple(std::make_pair("multiplier", multiplier)));
        sum = vent::reduce(outData.begin(), outData.end(), 0.0f);
        vent::transform(outData.begin(), outData.end(), outData.begin(), "float res(float a) {return a/sum;}", std::make_tuple(std::make_pair("sum", sum)));
    });

    std::vector<float> out2(n);
    std::transform(d1.begin(), d1.end(), out2.begin(), [](uint32_t v) { return float(v*v); });
    std::transform(out2.begin(), out2.end(), d1.begin(), out2.begin(), [multiplier](float a, uint32_t b) { return multiplier*(float(b) + a); });
    float sum2 = std::reduce(out2.begin(), out2.end(), 0.0f);
    std::transform(out2.begin(), out2.end(), out2.begin(), [sum](float a) { return a/sum; });


    for (int i = 0; i < n; i++) {
        if (out2[i] != 0) ASSERT_LT((outData[i] - out2[i])/out2[i], 0.001f);
    }

    if (sum > 0) ASSERT_LT((sum - sum2)/sum2, 0.001f);
    ASSERT_LT(std::reduce(outData.begin(), outData.end(), 0.0f) - 1.0f, 0.0001f);
}

TEST(VentTest, GpuRegionSpeedup) {
    const uint32_t n = 1'000'000;
    uint32_t iters = 10'000;
    std::vector<float> d1(n);
    for (int i = 0; i < n; ++i) {
        d1[i] = float(i);
    }
    std::vector<float> outData(n);

    auto start = std::chrono::high_resolution_clock::now();
    vent::gpu_region( vent::GpuRegionFlags::copyBuffersOut, [&]() {
        for (uint32_t i = 0; i < iters; i++) {
            vent::transform(d1.begin(), d1.end(), d1.begin(), "float res(float val) {return cos(2.0f * 3.14f * val/float(size));}");
        }
    });
    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "GPU Time: " << gpuTime << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (uint32_t i = 0; i < iters; i++) {
        for (uint32_t j = 0; j < n; j++) {
            d1[j] = std::cos(2.0f * 3.14f * d1[j]/float(n));
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "CPU Time: " << cpuTime << "s" << std::endl;
    std::cout << "Speedup: " << cpuTime / gpuTime << "x" << std::endl;
    std::cout << d1.back() << std::endl;

    ASSERT_GT(cpuTime / gpuTime, 1.0);
}

TEST(VentTest, SimpleJacobi) {
    uint32_t size = 3;
    std::vector<float> matrix = {
            5, -1, 2,
            3, 8, -2,
            1, 1, 4
    };

    std::vector<float> b = {12, -25, 6};

    std::vector<float> result(size, 0);

    vent::linsolve(matrix.begin(), matrix.end(), b.begin(), result.begin(), 200, 1e-4);

    ASSERT_LT(error(result[0], 1.0f), 1e-3);
    ASSERT_LT(error(result[1], -3.0f), 1e-3);
    ASSERT_LT(error(result[2], 2.0f), 1e-3);
}

TEST(VentTest, BigJacobi) {
    uint32_t size = 10'000;
    // create a diagonally dominant matrix
    std::vector<float> result(size, 0);
    for (auto& r : result) {
        r = randomFloat(-1, 1);
    }

    std::vector<float> matrix(size * size, 0);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) matrix[i * size + j] = randomFloat(float(size), float(size)*1.5f);
            else matrix[i * size + j] = randomFloat(-1, 1);
        }
    }

    std::vector<float> b(size, 0);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            b[i] += matrix[i * size + j] * result[j];
        }
    }

    std::vector<float> algoRes(size, 0);

    double epsilon = 1e-3;

    vent::transform(algoRes.begin(), algoRes.end(), algoRes.begin(), "float (float a) { return 0; }");

    auto start = std::chrono::high_resolution_clock::now();
    vent::linsolve(matrix.begin(), matrix.end(), b.begin(), algoRes.begin(), 200, epsilon);
    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "GPU Time: " << gpuTime << "ms" << std::endl;

    auto maxDiff = std::transform_reduce(result.begin(), result.end(), algoRes.begin(), 0.0f,
                                         [](float a, float b) {return std::max(a, std::abs(b));},
                                         [](float a, float b) {return std::abs(a - b);}
    );

//    for (int i = 0; i < size; i++) {
//        std::cout << result[i] << " " << algoRes[i] << std::endl;
//    }

    std::cout << "Max diff: " << maxDiff << std::endl;


    std::vector<float> result2(size, 0);
    std::vector<float> temp(size, 0);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < size; j++) {
            float sum = b[j];
            for (int k = 0; k < size; k++) {
                if (k != j) sum -= matrix[j * size + k] * temp[k];
            }
            result2[j] = sum / matrix[j * size + j];
        }

        if (i%10 == 0) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    if (j != k) sum += matrix[j * size + k] * result2[k];
                }
                temp[j] = (b[j] - (sum + matrix[j * size + j] * result2[j]))/b[j];
            }

            float error = std::abs(temp[0]);
            for (int j = 1; j < size; j++) {
                error = std::max(error, std::abs(temp[j]));
            }
//            std::cout << "Error: " << error << std::endl;

            if (error < epsilon) break;

        }
        result2.swap(temp);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "CPU Time: " << cpuTime << "ms" << std::endl;
    auto maxDiff2 = std::transform_reduce(result.begin(), result.end(), result2.begin(), 0.0f,
                                          [](float a, float b) {return std::max(a, std::abs(b));},
                                          [](float a, float b) {return std::abs(a - b);}
    );
    std::cout << "Max diff: " << maxDiff2 << std::endl;

    std::cout << "Speedup: " << cpuTime / gpuTime << "x" << std::endl;

    ASSERT_LT(error(maxDiff, maxDiff2), 0.01);
}

TEST(VentTest, NBody) {
//    uint32_t iter = 1'000;
//    uint32_t numParticles = 10'000;
    uint32_t iter = 10;
    uint32_t numParticles = 10;
    std::vector<float> initialPos(numParticles, 0);
    std::vector<float> initialVel(numParticles, 0);
    for (size_t i = 0; i < numParticles; i++) {
        initialPos[i] = randomFloat(-100, 100);
        initialVel[i] = randomFloat(-1, 1);
    }
    auto resetParticles = [numParticles, &initialPos, &initialVel](auto& p, auto& v) {
        for (size_t i = 0; i < numParticles; i++) {
            p[i] = initialPos[i];
            v[i] = initialVel[i];
        }
    };
    std::vector<float> positions(numParticles, 0);
    std::vector<float> velocities(numParticles, 0);
    resetParticles(positions, velocities);

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iter; i++) {
        vent::gpu_region(vent::GpuRegionFlags::keepBuffers | ((i == iter-1) ? vent::GpuRegionFlags::copyBuffersOut : vent::GpuRegionFlags::none),
                         [&](){
                             for (size_t i = 0; i < positions.size(); i++) {
                                 vent::transform(positions.begin(), positions.end(), velocities.begin(), velocities.begin(),
                                                 R"(float func(float p, float v) {
                                                float diff = pi - p;
                                                if (diff < 1e-5) return v;
                                                float invDist = 1.0 / abs(diff);
                                                float invDistCube = invDist * invDist * invDist;
                                                float s = 1.0f;
                                                return v + s * diff * invDistCube * 0.01f;
                                            }
                                            )", std::make_tuple(std::make_pair("pi", positions[i])));

                                 vent::transform(positions.begin(), positions.end(), velocities.begin(), positions.begin(),
                                                 R"(float func(float p, float v) {
                                                return p + v * 0.01f;
                                })");
                             }
                         });
    }
    auto end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "GPU Time: " << gpuTime << "s\n";

    resetParticles(positions, velocities);

    start = std::chrono::high_resolution_clock::now();
    for (uint32_t k = 0; k < iter; k++) {
        for (size_t i = 0; i < positions.size(); i++) {
            for (size_t j = 0; j < positions.size(); j++) {
                if (i == j) continue;
                float diff = positions[j] - positions[i];
                if (diff < 1e-5f) continue;
                float invDist = 1.0f / std::abs(diff);
                float invDistCube = invDist * invDist * invDist;
                float s = 1.0f;
                velocities[i] += s * diff * invDistCube * 0.01f;
            }
        }
        for (size_t i = 0; i < positions.size(); i++) {
            positions[i] += velocities[i] * 0.01f;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "CPU Time: " << cpuTime << "s\n";

    std::cout << "Speedup: " << cpuTime / gpuTime << "\n";
}

TEST(VentTest, SimpleCG) {
    auto pcg = []<typename T>(std::vector<T>& matrix, std::vector<T>& b, std::vector<T>& res, T epsilon) {
        auto dot = [](const std::vector<T>& a, const std::vector<T>& b) {
            T sum = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                sum += a[i]*b[i];
            }
            return sum;
        };

        res.resize(b.size(), 0);
        std::vector<T> r = b;
        std::vector<T> s = b;

        std::vector<T> as(b.size(), 0);

        auto initialDot = dot(r, r);

        for (uint32_t iter = 0; iter < 200; iter++) {
            // as = A*s
            for (size_t i = 0; i < as.size(); i++) {
                as[i] = 0;
                for (size_t j = 0; j < as.size(); j++) {
                    as[i] += matrix[i * as.size() + j] * s[j];
                }
            }

            T sas = dot(s, as);
            T rr = dot(r, r);

            if (rr < epsilon*initialDot) {
                break;
            }

            T alpha = rr / sas;

            for (size_t i = 0; i < res.size(); ++i) res[i] = res[i] + alpha*s[i];

            for (size_t i = 0; i < r.size(); ++i) r[i] = r[i] - alpha * as[i];

            T prevRr = rr;
            rr = dot(r, r);
            T beta = rr / prevRr;

            // s = r + beta*s
            for (size_t i = 0; i < s.size(); ++i) s[i] = r[i] + beta*s[i];

        }

    };

    uint32_t size = 2;
    std::vector<float> matrix = {
            4, 1,
            1, 3
    };
    std::vector<float> vector = {1, 2};

    std::vector<float> results2(size);
    std::vector<float> results3(size);

    vent::linsolve(matrix.begin(), matrix.end(), vector.begin(), results2.begin(), 200, 1e-4, true);
    pcg(matrix, vector, results3, 1e-4f);

    for (uint32_t i = 0; i < size; i++) {
        ASSERT_LT(error(results2[i], results3[i]), 0.0001f);
    }

}

template<typename T>
void matrixMultiply(const std::vector<T>& matrix, const std::vector<T>& vector, std::vector<T>& result) {
    for (uint32_t i = 0; i < result.size(); ++i) {
        result[i] = 0;
        for (uint32_t j = 0; j < result.size(); ++j) {
            result[i] += matrix[i * result.size() + j] * vector[j];
        }
    }
}

template<typename T>
void pcg(std::vector<T>& matrix, std::vector<T>& b, std::vector<T>& res) {
    auto dot = [](const std::vector<T>& a, const std::vector<T>& b) {
        T sum = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i]*b[i];
        }
        return sum;
    };

    res.resize(b.size(), 0);
    std::vector<T> r = b;
    std::vector<T> s = b;

    std::vector<T> as(b.size(), 0);

    auto initialDot = dot(r, r);

    for (uint32_t iter = 0; iter < 200; iter++) {
        // as = A*s
        matrixMultiply(matrix, s, as);
        T sas = dot(s, as);
        T rr = dot(r, r);

        if (rr < 1e-6*initialDot) {
            std::cout << "num iterations: " << iter << "\n";
            break;
        }

        T alpha = rr / sas;

        for (size_t i = 0; i < res.size(); ++i) res[i] = res[i] + alpha*s[i];

        for (size_t i = 0; i < r.size(); ++i) r[i] = r[i] - alpha * as[i];

        T prevRr = rr;
        rr = dot(r, r);
        T beta = rr / prevRr;

        // s = r + beta*s
        for (size_t i = 0; i < s.size(); ++i) s[i] = r[i] + beta*s[i];

    }

}

std::vector<float> createOrthogonalMatrix(uint32_t size) {
    static auto norm = [](std::vector<float>& x, uint32_t start, uint32_t end) {
        float s = 0;
        for (uint32_t i = start; i < end; i++ )
        {
            s += x[i] * x[i];
        }
        return std::sqrt(s);
    };

    static auto sign = [](float x) {
        if (x < 0) return -1.0f;
        if (x > 0) return 1.0f;
        return 0.0f;
    };

    std::vector<float> matrix(size*size, 0);
    std::vector<float> v(size, 0), x(size, 0);

    for (uint32_t i = 0; i < size; i++ )
    {
        matrix[i+i*size] = 1.0;
    }

    for (uint32_t j = 0; j < size - 1; j++) {
//
//  Set the vector that represents the J-th column to be annihilated.
//
        for (uint32_t i = 0; i < j; i++ ) {
            x[i] = 0.0;
        }
        for (uint32_t i = j; i < size; i++ ) {
            x[i] = randomFloat();
        }

        uint32_t k = j+1;
        if (k > 1 && size > k) {
            float s = norm(x, k-1, size - k + 1);

            if (s != 0.0) {
                v[k - 1] = x[k - 1] + std::fabs(s) * sign(x[k - 1]);

                for (uint32_t i = k; i < size - k; i++) {
                    v[i] = x[i];
                }

                s = norm(v, k-1, size - k + 1);
                if (s != 0.0f) {
                    for (uint32_t i = k - 1; i < size; i++) {
                        v[i] = v[i] / s;
                    }
                }
            }
        }

        float v_normsq = 0.0f;
        for (uint32_t i = 0; i < size; i++){
            v_normsq += v[i] * v[i];
        }

        if (v_normsq == 0.0) {
            continue;
        }
        std::vector<float> ah(size*size);

        for (uint32_t m = 0; m < size; m++ )
        {
            for (uint32_t i = 0; i < size; i++ )
            {
                ah[i+m*size] = matrix[i+m*size];
                for ( k = 0; k < size; k++ )
                {
                    ah[i+m*size] = ah[i+m*size] - 2.0f * matrix[i+k*size] * v[k] * v[j] / v_normsq;
                }
            }
        }

        for (uint32_t m = 0; m < size; m++){
            for (uint32_t i = 0; i < size; i++){
                matrix[i+m*size] = ah[i+m*size];
            }
        }
    }

    return matrix;

}

std::vector<float> createPdsMatrix(uint32_t size) {
    std::vector<float> matrix(size * size);
    std::vector<float> lambda(size);
    std::vector<float> q(size);
//
//  Get a random set of eigenvalues.
//
    for (float & l : lambda) l = randomFloat();
//
//  Get a random orthogonal matrix Q.
//
    q = createOrthogonalMatrix(size);
//
//  Set A = Q * Lambda * Q'.
//
    for (uint32_t j = 0; j < size; j++) {
        for (uint32_t i = 0; i < size; i++) {
            matrix[i+j*size] = 0.0;
            for (uint32_t k = 0; k < size; k++) {
                matrix[i+j*size] += q[i+k*size] * lambda[k] * q[j+k*size];
            }
        }
    }

    return matrix;
}

TEST(VentTest, MediumCG) {
    // Function to generate a positive definite matrix
    uint32_t size = 10;
    std::vector<float> matrix = createPdsMatrix(size);

    std::cout << "Created matrix\n";

    // Generate a random vector
    std::vector<float> results(size);
    for (int i = 0; i < size; ++i) {
        results[i] = randomFloat(-1.0f, 1.0f);
    }

    // Multiply the matrix by the vector
    std::vector<float> vector(size);
    for (int i = 0; i < size; ++i) {
        float sum = 0;
        for (int j = 0; j < size; ++j) {
            sum += matrix[i * size + j] * results[j];
        }
        vector[i] = sum;
    }

    std::vector<float> results2(size);

    vent::linsolve(matrix.begin(), matrix.end(), vector.begin(), results2.begin(), 200, 1e-6, true);

    std::vector<float> results3(size);

    pcg(matrix, vector, results3);


    for (uint32_t i = 0; i < size; i++) {
        ASSERT_LT(error(results2[i], results3[i]), 0.001f);
        ASSERT_LT(error(results[i], results2[i]), 0.02f);
    }
}

