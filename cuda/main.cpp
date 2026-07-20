#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <cmath>

#define N 30
#define NUM_CENTROS 3

extern "C" void lanzarMatVecMul(const float *h_A, const float *h_B, float *h_C, int n);
extern "C" void resolver_cuda(const float *ax,
                              const float *ay,
                              const float *az,
                              const float *cx,
                              const float *cy,
                              const float *cz,
                              int *indices_cuda,
                              int n);

int indice_mas_cercano_serial(float px, float py, float pz,
                              const std::vector<float> &cx,
                              const std::vector<float> &cy,
                              const std::vector<float> &cz)
{
    float mejor_dist =
        (px - cx[0]) * (px - cx[0]) +
        (py - cy[0]) * (py - cy[0]) +
        (pz - cz[0]) * (pz - cz[0]);

    int mejor_indice = 0;

    for (int i = 1; i < NUM_CENTROS; i++)
    {
        float dist =
            (px - cx[i]) * (px - cx[i]) +
            (py - cy[i]) * (py - cy[i]) +
            (pz - cz[i]) * (pz - cz[i]);

        if (dist < mejor_dist)
        {
            mejor_dist = dist;
            mejor_indice = i;
        }
    }

    return mejor_indice;
}

void resolver_serial(const std::vector<float> &ax,
                     const std::vector<float> &ay,
                     const std::vector<float> &az,
                     const std::vector<float> &cx,
                     const std::vector<float> &cy,
                     const std::vector<float> &cz,
                     std::vector<int> &indices_serial)
{
    for (int i = 0; i < N; i++)
    {
        indices_serial[i] = indice_mas_cercano_serial(ax[i], ay[i], az[i], cx, cy, cz);
    }
}

int main()
{
    const int n = 25;

    std::vector<float> h_A(n * n, 0.0);
    std::vector<float> h_B(n, 1.0);
    std::vector<float> h_C(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            h_A[i * n + j] = i + 1;
        }
    }

    lanzarMatVecMul(h_A.data(), h_B.data(), h_C.data(), n);

    for (int i = 0; i < n; i++)
        std::cout << h_C[i] << " ";

    std::vector<float> ax(N), ay(N), az(N);
    std::vector<float> cx(NUM_CENTROS), cy(NUM_CENTROS), cz(NUM_CENTROS);
    std::vector<int> indices_serial(N), indices_cuda(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 10);

    for (int i = 0; i < N; i++)
    {
        ax[i] = (float)dist(gen);
        ay[i] = (float)dist(gen);
        az[i] = (float)dist(gen);
    }

    for (int i = 0; i < NUM_CENTROS; i++)
    {
        cx[i] = (float)dist(gen);
        cy[i] = (float)dist(gen);
        cz[i] = (float)dist(gen);
    }

    resolver_serial(ax, ay, az, cx, cy, cz, indices_serial);

    resolver_cuda(ax.data(), ay.data(), az.data(),
                  cx.data(), cy.data(), cz.data(),
                  indices_cuda.data(), N);

    std::cout << "\nCentros:\n";
    for (int i = 0; i < NUM_CENTROS; i++)
    {
        std::cout << "(" << cx[i] << "," << cy[i] << "," << cz[i] << ")\n";
    }

    std::cout << "\n";
    std::cout << "Vector\tIndice serial\tIndice paralelo\n";

    for (int i = 0; i < N; i++)
    {
        std::cout << "(" << (int)ax[i] << "," << (int)ay[i] << "," << (int)az[i] << ") "
                  << indices_serial[i] << " "
                  << indices_cuda[i] << "\n";
    }

    return 0;
}