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

int main1() {
    const uint32_t cells_per_line = 120;

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
    auto seq_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Time taken using a single thread: " << seq_time << " seconds\n";
    std::fill(res.begin(), res.end(), 0);
    start = std::chrono::high_resolution_clock::now();
    vent::linsolve(m.begin(), m.end(), b.begin(), res.begin(), 200, 1e-6, true);
    end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<double>(end - start).count();
    std::cout << "Time taken using vent: " << time << " seconds, speedup: " << seq_time/time << "\n";
}

int main() {

}

