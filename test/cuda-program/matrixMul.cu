/**
 * Matrix multiplication: C = A * B
 * This sample demonstrates basic shared memory optimizations for matrix multiplication
 * From CUDA SDK Samples
 */

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    
    float Cvalue = 0;
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        
        __syncthreads();
        
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        
        __syncthreads();
    }
    
    SetElement(Csub, row, col, Cvalue);
}

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main(int argc, char **argv)
{
    int block_size = BLOCK_SIZE;
    int width = 5 * block_size;
    int height = 5 * block_size;
    
    Matrix A, B, C;
    A.width = width;
    A.height = height;
    A.elements = (float*)malloc(width * height * sizeof(float));
    
    B.width = width;
    B.height = height;
    B.elements = (float*)malloc(width * height * sizeof(float));
    
    C.width = width;
    C.height = height;
    C.elements = (float*)malloc(width * height * sizeof(float));
    
    for(int i = 0; i < width * height; i++) {
        A.elements[i] = 1.0f;
        B.elements[i] = 2.0f;
    }
    
    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", A.height, A.width, B.height, B.width);
    
    MatMul(A, B, C);
    
    bool correct = true;
    float valB = width * 2.0f;
    for (int i = 0; i < width * height; i++) {
        if (fabs(C.elements[i] - valB) > 0.001f) {
            printf("Error at C[%d] = %f, expected %f\n", i, C.elements[i], valB);
            correct = false;
            break;
        }
    }
    
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
    
    free(A.elements);
    free(B.elements);
    free(C.elements);
    
    return 0;
}