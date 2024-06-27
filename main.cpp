#include <iostream>
#include <chrono>
#include <random>
#include "vent.h"

inline float randomFloat(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis_float;
    return dis_float(gen);
}

inline float randomFloat(float min, float max) {
    return min + randomFloat() * (max - min);
}

void multipleTransformsSpeedup() {
    const uint32_t n = 10'000'000;
    uint32_t iters = 10'000;
    std::vector<float> d1(n);
    std::vector<float> d2(n);
    std::vector<float> d3(n);

    for (int i = 0; i < n; ++i) {
        d1[i] = float(i);
    }
    std::vector<float> outData(n);

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iters; i++) {
        for (uint32_t j = 0; j < n; j++) {
            d1[j] = std::cos(2.0f * 3.14f * d1[j]/float(n));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "CPU Time: " << cpuTime << "s" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    vent::gpu_region( vent::GpuRegionFlags::copyBuffersOut, [&]() {
        for (uint32_t i = 0; i < iters; i++) {
            vent::transform(d2.begin(), d2.end(), d2.begin(), "float res(float val) {return cos(2.0f * 3.14f * val/float(size));}");
        }
    });
    end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "GPU Time: " << gpuTime << "s" << std::endl;
    std::cout << "Speedup: " << cpuTime / gpuTime << "x" << std::endl;


    start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (uint32_t i = 0; i < iters; i++) {
        for (uint32_t j = 0; j < n; j++) {
            d3[j] = std::cos(2.0f * 3.14f * d1[j]/float(n));
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double ompTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "OMP Time: " << ompTime << "s" << std::endl;
    std::cout << "Speedup: " << cpuTime / ompTime << "x" << std::endl;

    std::cout << d1.back() << std::endl;
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

//        # pragma omp parallel for
        for (size_t i = 0; i < res.size(); ++i) res[i] = res[i] + alpha*s[i];

//        # pragma omp parallel for
        for (size_t i = 0; i < r.size(); ++i) r[i] = r[i] - alpha * as[i];

        T prevRr = rr;
        rr = dot(r, r);
        T beta = rr / prevRr;

        // s = r + beta*s
//        # pragma omp parallel for
        for (size_t i = 0; i < s.size(); ++i) s[i] = r[i] + beta*s[i];

    }

}

template<typename T>
void ompMatrixMultiply(const std::vector<T>& matrix, const std::vector<T>& vector, std::vector<T>& result) {
# pragma omp parallel for
    for (uint32_t i = 0; i < result.size(); ++i) {
        result[i] = 0;
        for (uint32_t j = 0; j < result.size(); ++j) {
            result[i] += matrix[i * result.size() + j] * vector[j];
        }
    }
}

template<typename T>
void ompPcg(std::vector<T>& matrix, std::vector<T>& b, std::vector<T>& res) {
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

        # pragma omp parallel for
        for (size_t i = 0; i < res.size(); ++i) res[i] = res[i] + alpha*s[i];

        # pragma omp parallel for
        for (size_t i = 0; i < r.size(); ++i) r[i] = r[i] - alpha * as[i];

        T prevRr = rr;
        rr = dot(r, r);
        T beta = rr / prevRr;

        // s = r + beta*s
        # pragma omp parallel for
        for (size_t i = 0; i < s.size(); ++i) s[i] = r[i] + beta*s[i];

    }

}

void cgSpeedup() {
    const uint32_t cells_per_line = 140;

    std::vector<float> solidCells(cells_per_line*cells_per_line);
    for (uint32_t i = 0; i < cells_per_line; ++i) {
        for (uint32_t j = 0; j < cells_per_line; ++j) {
            solidCells[i + cells_per_line*j] = (i == 0 || j == 0 || i == cells_per_line - 1 || j == cells_per_line - 1) ? 0 : 1;
        }
    }

    std::vector<float> m(solidCells.size()*solidCells.size());

    for (uint32_t i = 1; i < solidCells.size(); ++i) {
        if (i < cells_per_line ||
            i % cells_per_line == 0 ||
            i % cells_per_line == cells_per_line - 1 ||
            i >= solidCells.size() - cells_per_line)
            continue;

        uint total = 0;
        uint typeRight = (uint) solidCells[i+1];
        m[i*solidCells.size() + i+1] = (typeRight == 1) ? -1 : 0;
        total += uint(typeRight != 0);

        uint typeLeft = (uint) solidCells[i-1];
        m[i*solidCells.size() + i-1] = (typeLeft == 1) ? -1 : 0;
        total += uint(typeLeft != 0);

        uint typeForward = (uint) solidCells[i+cells_per_line];
        m[i*solidCells.size() + i+cells_per_line] = (typeForward == 1) ? -1 : 0;
        total += uint(typeForward != 0);

        uint typeBackward = (uint) solidCells[i-cells_per_line];
        m[i*solidCells.size() + i-cells_per_line] = (typeBackward == 1) ? -1 : 0;
        total += uint(typeBackward != 0);

        m[i*solidCells.size() + i] = float(total);
    }


    std::vector<float> b(solidCells.size());
    for (size_t i = 0; i < b.size(); ++i) {
        if (solidCells[i] == 0) continue;
        b[i] = float(i);
    }
    std::vector<float> res(solidCells.size(), 0);

    std::cout << "Solving a linear system using the CG method\n";

    auto start = std::chrono::high_resolution_clock::now();
    pcg(m, b, res);
    auto end = std::chrono::high_resolution_clock::now();

    auto seqTime = std::chrono::duration<double>(end - start).count();
    std::cout << "Time taken using a single thread: " << seqTime << " seconds\n";

    std::fill(res.begin(), res.end(), 0);

    start = std::chrono::high_resolution_clock::now();
    vent::linsolve(m.begin(), m.end(), b.begin(), res.begin(), 200, 1e-6, true);
    end = std::chrono::high_resolution_clock::now();

    auto ompTime = std::chrono::duration<double>(end - start).count();

    std::cout << "Time taken using omp: " << ompTime << " seconds, speedup: " << seqTime/ompTime << "\n";


    std::fill(res.begin(), res.end(), 0);

    start = std::chrono::high_resolution_clock::now();
    vent::linsolve(m.begin(), m.end(), b.begin(), res.begin(), 200, 1e-6, true);
    end = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration<double>(end - start).count();

    std::cout << "Time taken using vent: " << gpuTime << " seconds, speedup: " << seqTime/gpuTime << "\n";
}

void nBodySpeedup() {
    uint32_t iter = 1'000;
    uint32_t numParticles = 1'000;
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
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "CPU Time: " << cpuTime << "s\n";

    resetParticles(positions, velocities);

    start = std::chrono::high_resolution_clock::now();
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
    end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "GPU Time: " << gpuTime << "s\n";
    std::cout << "Speedup: " << cpuTime / gpuTime << "\n";

    resetParticles(positions, velocities);

    start = std::chrono::high_resolution_clock::now();
    for (uint32_t k = 0; k < iter; k++) {
        #pragma omp parallel for
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
        #pragma omp parallel for
        for (size_t i = 0; i < positions.size(); i++) {
            positions[i] += velocities[i] * 0.01f;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double ompTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "OMP Time: " << ompTime << "s\n";

    std::cout << "Speedup: " << cpuTime / ompTime << "\n";
}

int main() {
    nBodySpeedup();
    return 0;
}

