#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(err);                                                 \
        }                                                              \
    } while (0)

__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float val = 0;

    for (int x = 0; x < (N + TILE_SIZE - 1)/TILE_SIZE; ++x) {

        if (row < N && (x*TILE_SIZE + tx) < N)
            A_tile[ty][tx] = A[row * N + x * TILE_SIZE + tx];
        else
            A_tile[ty][tx] = 0.0f;

        if (col < N && (x*TILE_SIZE + ty) < N)
            B_tile[ty][tx] = B[(x * TILE_SIZE + ty) * N + col];
        else
            B_tile[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            val += A_tile[ty][k] * B_tile[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = val;
}

int main() {
    int N = 1024;  
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for(int i=0;i<N*N;i++) {
        h_A[i] = 1.0f; 
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    printf("C[0] = %f\n", h_C[0]); 

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
