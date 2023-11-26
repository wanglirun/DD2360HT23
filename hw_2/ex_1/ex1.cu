%%writefile vectorAdd.cu

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#define DataType double
#define TPB 64

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
	const int idx=threadIdx.x+blockDim.x*blockIdx.x;
	if(idx>=len) return;
	out[idx]=in1[idx]+in2[idx];
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if(argc<2) printf("input length of the vector please!");
  else inputLength=atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
resultRef = (DataType *)malloc(inputLength * sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  //curandState dev_random;
  //curand_init(0, 0, 0, &dev_random);
  srand(time(NULL));
  for(int i=0;i<inputLength;++i)
  {
	hostInput1[i]=(double)rand() / RAND_MAX;
	hostInput2[i]=(double)rand() / RAND_MAX;
	resultRef[i]=hostInput1[i]+hostInput2[i];
  }
  

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1,inputLength*sizeof(DataType));
  cudaMalloc(&deviceInput2,inputLength*sizeof(DataType));
  cudaMalloc(&deviceOutput,inputLength*sizeof(DataType));	

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1,hostInput1,inputLength*sizeof(DataType),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2,hostInput2,inputLength*sizeof(DataType),cudaMemcpyHostToDevice);
  
  //@@ Initialize the 1D grid and block dimensions here
  

  //@@ Launch the GPU Kernel here
  vecAdd<<<(inputLength+TPB-1)/TPB,TPB>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);

cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput,deviceOutput,inputLength*sizeof(DataType),cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  long double tmp=0;
  for(int i=0;i<inputLength;++i)
  {
	tmp+=abs(resultRef[i]-hostOutput[i]);
  }
  tmp/=inputLength;
  printf("average difference:%LF\n",tmp);
  //@@ Free the GPU memory here
free(hostInput1);
free(hostInput2);
free(hostOutput);
free(resultRef);
  //@@ Free the CPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  return 0;
}

