%%writefile gemm.cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define TPB 32

#define DataType double

// GPU Kernel to compute matrix multiplication
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < numARows) && (Col < numBColumns)) {
        DataType Cvalue = 0;
        for (int k = 0; k < numAColumns; ++k) {
            Cvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
        }
        C[Row * numBColumns + Col] = Cvalue;
    }
}

int main(int argc, char **argv) {
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    // Read in numARows, numAColumns, numBColumns from args
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <numARows> <numAColumns> <numBColumns>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBColumns = atoi(argv[3]);
    numBRows = numAColumns; // Since A's columns should match B's rows
    numCRows = numARows;
    numCColumns = numBColumns;

    printf("Input matrix dimensions: (%d x %d), (%d x %d), (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    // Allocate Host memory
    hostA = (DataType *)malloc(sizeof(DataType) * numARows * numAColumns);
    hostB = (DataType *)malloc(sizeof(DataType) * numBRows * numBColumns);
    hostC = (DataType *)malloc(sizeof(DataType) * numCRows * numCColumns);
    resultRef = (DataType *)malloc(sizeof(DataType) * numCRows * numCColumns);

    // Initialize hostA and hostB with random values
    // Initialize resultRef with zeroes or appropriate values
    // [Code to initialize hostA, hostB, and resultRef]

    // Allocate GPU memory
    cudaMalloc((void **)&deviceA, sizeof(DataType) * numARows * numAColumns);
    cudaMalloc((void **)&deviceB, sizeof(DataType) * numBRows * numBColumns);
    cudaMalloc((void **)&deviceC, sizeof(DataType) * numCRows * numCColumns);

    // Copy memory to the GPU
    cudaMemcpy(deviceA, hostA, sizeof(DataType) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(DataType) * numBRows * numBColumns, cudaMemcpyHostToDevice);

    // Initialize the grid and block dimensions
    dim3 dimGrid(ceil(numCColumns/TPB), ceil(numCRows/TPB), 1);
    dim3 dimBlock(TPB,TPB, 1);

    // Launch the GPU Kernel
    gemm<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

    // Copy the GPU memory back to the CPU
    cudaMemcpy(hostC, deviceC, sizeof(DataType) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // Free CPU memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);

    return 0;
}
