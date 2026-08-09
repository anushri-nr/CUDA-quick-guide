#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixTranspose(const float *input, float *output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

int main() {
    int rows = 3;
    int cols = 2;
    float h_input[rows * cols] = {1, 2, 3, 4, 5, 6}; // Example input matrix
    float h_output[rows * cols]; // Output matrix

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            printf("output[%d][%d] = %f\n", i, j, h_output[i * rows + j]);
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
}