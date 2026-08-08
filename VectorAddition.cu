#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 10;
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}