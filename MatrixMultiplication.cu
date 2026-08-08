#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMultiplication(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += A[row * N + n] * B[n * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main() {
    int M = 2; // Number of rows in A and C
    int N = 3; // Number of columns in A and rows in B
    int K = 4; // Number of columns in B and C

    float h_A[M * N] = {1, 2, 3, 4, 5, 6}; // Example matrix A
    float h_B[N * K] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}; // Example matrix B
    float h_C[M * K]; // Result matrix C

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("C[%d][%d] = %f\n", i, j, h_C[i * K + j]);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}