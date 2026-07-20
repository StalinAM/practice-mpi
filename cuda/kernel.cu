#include <cuda_runtime.h>
#include <cmath>

__global__ void matVecMul(const float* A, const float* B, float* C, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += A[tid * n + k] * B[k];
        }
        C[tid] = sum;
    }
}

extern "C" void lanzarMatVecMul(const float* h_A, const float* h_B, float* h_C, int n)
{
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (int)ceil((double)n / blockSize);

    matVecMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#define NUM_CENTROS 3

__device__ int indice_mas_cercano_cuda(float px, float py, float pz,
                                       const float* cx,
                                       const float* cy,
                                       const float* cz)
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

__global__ void kernel_centros_mas_cercanos(const float* ax,
                                            const float* ay,
                                            const float* az,
                                            const float* cx,
                                            const float* cy,
                                            const float* cz,
                                            int* indices,
                                            int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        indices[tid] = indice_mas_cercano_cuda(ax[tid], ay[tid], az[tid], cx, cy, cz);
    }
}

extern "C" void resolver_cuda(const float* ax,
                              const float* ay,
                              const float* az,
                              const float* cx,
                              const float* cy,
                              const float* cz,
                              int* indices_cuda,
                              int n)
{
    float *d_ax, *d_ay, *d_az;
    float *d_cx, *d_cy, *d_cz;
    int *d_indices;

    cudaMalloc((void**)&d_ax, n * sizeof(float));
    cudaMalloc((void**)&d_ay, n * sizeof(float));
    cudaMalloc((void**)&d_az, n * sizeof(float));

    cudaMalloc((void**)&d_cx, NUM_CENTROS * sizeof(float));
    cudaMalloc((void**)&d_cy, NUM_CENTROS * sizeof(float));
    cudaMalloc((void**)&d_cz, NUM_CENTROS * sizeof(float));

    cudaMalloc((void**)&d_indices, n * sizeof(int));

    cudaMemcpy(d_ax, ax, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay, ay, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_az, az, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_cx, cx, NUM_CENTROS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, cy, NUM_CENTROS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cz, cz, NUM_CENTROS * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (int)ceil((double)n / blockSize);

    kernel_centros_mas_cercanos<<<gridSize, blockSize>>>(d_ax, d_ay, d_az,
                                                         d_cx, d_cy, d_cz,
                                                         d_indices, n);

    cudaDeviceSynchronize();

    cudaMemcpy(indices_cuda, d_indices, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_cz);
    cudaFree(d_indices);
}