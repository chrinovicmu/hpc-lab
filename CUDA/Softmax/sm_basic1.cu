
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(float *input, float* output, int num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num) return;

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = -INFINITY;
    __syncthreads();

    atomicMax((int*)&shared_max, __float_as_int(input[tid]));
    __syncthreads();

    float max_val = shared_max;

    float exp_val = expf(input[tid] - max_val);

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();

    atomicAdd(&shared_sum, exp_val);
    __syncthreads();

    float sum_exp = shared_sum;

    output[tid] = exp_val / sum_exp;
}

int main()
{
    const int N = 8;
    float h_input[N]  = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float h_output[N] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 8;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Softmax output:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_output[i] << " ";
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
