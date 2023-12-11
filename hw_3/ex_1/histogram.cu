#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define MAX_BIN_VAL 127
#define THREADS_PER_BLOCK 256

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    extern __shared__ unsigned int histo_private[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (i < num_elements) {
        atomicAdd(&(histo_private[input[i]]), 1);
        i += offset;
    }
    __syncthreads();

    // Write the results to the global memory histogram
    i = threadIdx.x;
    while (i < num_bins) {
        atomicAdd(&(bins[i]), histo_private[i]);
        i += blockDim.x;
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_bins) {
        bins[i] = min(bins[i], MAX_BIN_VAL);
    }
}

int main(int argc, char **argv) {
  
  int inputLength = atoi(argv[1]); // Assuming the first argument is the input length
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  printf("The input length is %d\n", inputLength);
  
  // Allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  // Initialize hostInput with random numbers in the range [0, NUM_BINS-1]
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, NUM_BINS - 1);
  for (int i = 0; i < inputLength; ++i) {
    hostInput[i] = dis(gen);
  }

  // Initialize hostBins with zeros
  for (int i = 0; i < NUM_BINS; ++i) {
    hostBins[i] = 0;
  }

  // Allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  // Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  // Initialize the grid and block dimensions here
  dim3 dimGrid((inputLength - 1) / THREADS_PER_BLOCK + 1, 1, 1);
  dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

  // Launch the GPU Kernel here
  histogram_kernel<<<dimGrid, dimBlock, NUM_BINS * sizeof(unsigned int)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  // Initialize the second grid and block dimensions here
  // For convert_kernel, we only need as many threads as there are bins
  dim3 c_dimGrid((NUM_BINS - 1) / THREADS_PER_BLOCK + 1, 1, 1);
  dim3 c_dimBlock(THREADS_PER_BLOCK, 1, 1);

  // Launch the second GPU Kernel here
  convert_kernel<<<c_dimGrid, c_dimBlock>>>(deviceBins, NUM_BINS);

  // Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // Compare the output with the reference, validate and print the results

  // Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  // Free the CPU memory here
  free(hostInput);
  free(hostBins);

  return 0;
}
