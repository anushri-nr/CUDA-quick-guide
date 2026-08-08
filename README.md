# CUDA Quick Guide

A short overview of GPU architecture and how to approach CUDA programming.

## CPU vs. GPU

A CPU has a small number of powerful, general-purpose cores optimized for low-latency execution of a wide variety of instructions. A GPU has a much larger number of simpler cores optimized for running the same instruction across many data elements in parallel.

This makes GPUs well suited to workloads that are "embarrassingly parallel" — where the same operation is applied independently to many pieces of data (vector/matrix math, image processing, simulations). CUDA is NVIDIA's programming model for writing code that runs on the GPU (the "device") and is launched from code running on the CPU (the "host").

Example specs (NVIDIA RTX 3090):

- 10,496 CUDA cores
- 82 Streaming Multiprocessors (SMs)
- Up to 2,048 threads resident per SM
- ~170,000 threads resident concurrently across the GPU
- 24 GB global memory
- 128 KB shared memory per SM

## The CUDA execution hierarchy

CUDA organizes parallel work into three levels:

1. **Thread** — the basic unit of execution; runs one instance of a function called a **kernel**.
2. **Block** — a group of threads that run on the same SM and can cooperate via shared memory and `__syncthreads()`.
3. **Grid** — all the blocks launched for a single kernel call.

A kernel launch specifies the grid and block dimensions (1D, 2D, or 3D), and each thread uses built-in variables to figure out which piece of data it's responsible for:

- `threadIdx` — the thread's index within its block
- `blockIdx` — the block's index within the grid
- `blockDim` — the number of threads per block

```
Grid
 ├── Block(0,0)   Block(0,1)
 └── Block(1,0)   Block(1,1)
       └── each Block is a 1D/2D/3D arrangement of Threads
```

## Memory hierarchy

Performance in CUDA is largely about where data lives and how often it has to move.

| Memory | Scope | Bandwidth | Size | Latency |
| --- | --- | --- | --- | --- |
| Global memory | All threads, all blocks | ~10-50 GB/s | 8-80 GB total | ~400-800 cycles |
| Shared memory | Threads within one block | ~1-10 TB/s | 48-164 KB per SM | ~20-30 cycles |
| Registers | Single thread | ~19 TB/s | 64 KB per SM | ~1 cycle |

- **Global memory** is the GPU's main VRAM. Any thread can read/write it, but it's relatively slow.
- **Shared memory** is a small, fast, per-SM scratchpad visible only to threads in the same block — useful for avoiding redundant global memory accesses when threads in a block need overlapping data.
- **Registers** are per-thread and fastest, but limited in number.

Host (CPU) and device (GPU) also have separate memory spaces, connected over PCIe:

| Connection | Typical bandwidth |
| --- | --- |
| PCIe 3.0 x16 | ~10 GB/s |
| PCIe 4.0 x16 | ~22 GB/s |
| PCIe 5.0 x16 | ~45 GB/s |

Host-device transfers are much slower than on-device memory access, so CUDA programs generally try to minimize transfers, keep data resident on the GPU across multiple kernel launches, and overlap transfer with computation where possible.

## The shape of a CUDA program

Most CUDA programs, regardless of what the kernel computes, follow the same host-side pattern:

1. **Allocate** device memory (`cudaMalloc`).
2. **Copy** input data from host to device (`cudaMemcpy`, `cudaMemcpyHostToDevice`).
3. **Launch** the kernel with a grid/block configuration (`kernel<<<blocks, threads>>>(...)`).
4. **Copy** results back from device to host (`cudaMemcpy`, `cudaMemcpyDeviceToHost`).
5. **Free** device memory (`cudaFree`).

The kernel itself is a `__global__` function: each thread computes its global index from `threadIdx`/`blockIdx`/`blockDim`, then operates on the data at that index.

## How to approach writing a CUDA kernel

1. **Find the parallelism.** Identify the unit of work that can be computed independently (one output element, one row, one pixel) and map one thread to one unit.
2. **Get a correct, naive version working first.** Don't worry about memory access patterns yet — just get each thread reading its inputs from global memory and writing its output.
3. **Profile before optimizing.** Naive kernels are often memory-bound (spending most of their time waiting on global memory) rather than compute-bound. Tools like `nvprof`/Nsight Compute will show this.
4. **Reduce redundant global memory traffic.** If multiple threads in a block read the same global memory, consider having the block cooperatively load that data into shared memory once and reuse it, synchronizing with `__syncthreads()` between the load and the compute phases.
5. **Check for coalesced access.** Threads in a warp accessing contiguous memory addresses are much more efficient than scattered access patterns.
6. **Reach for existing libraries when possible.** For common operations (linear algebra, FFTs, deep learning primitives), cuBLAS, cuFFT, and cuDNN are heavily tuned and usually outperform hand-written kernels.

## Compiling and running a CUDA program

CUDA source files use the `.cu` extension (host and device code can live in the same file). Header files use `.cuh` by convention.

Compilation and execution go through `nvcc`, NVIDIA's CUDA compiler, which splits the file into host code (handed off to your system C++ compiler) and device code (compiled for the GPU):

```
# Compile
nvcc program.cu -o program

# Run
./program
```

Check that the toolkit is installed and see which compute capability your GPU supports:

```
nvcc --version       # confirm nvcc is installed and check CUDA version
nvidia-smi            # confirm the driver sees the GPU
```

## Further reading

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
