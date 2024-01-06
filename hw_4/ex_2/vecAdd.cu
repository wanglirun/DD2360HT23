%%writefile vectorAdd.cu

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#define DataType double
#define TPB 1024
//#define N_seg 8 //divide every inputlength/4 elements in a stream into S_seg parts.
#define DEBUG 0
using namespace std;
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len,int offset) {
  //@@ Insert code to implement vector addition here
	const int idx=threadIdx.x+blockDim.x*blockIdx.x+offset;
  if(DEBUG) printf("%d ",idx);
	if(idx>=len) return;
	out[idx]=in1[idx]+in2[idx];
  if(DEBUG) printf("%lf\n",in1[idx]);
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop


int main(int argc, char **argv) {

  int inputLength,N_seg;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  DataType *resultRef;

  //@@ Insert code below to read in inputLength from args
  if(argc<3) printf("input length of the vector please!");
  else {
    inputLength=atoi(argv[1]);
    N_seg=atoi(argv[2]);
  }

  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  cudaHostAlloc((void**)&hostInput1,inputLength * sizeof(DataType),cudaHostAllocDefault);
  cudaHostAlloc((void**)&hostInput2,inputLength * sizeof(DataType),cudaHostAllocDefault);
  cudaHostAlloc((void**)&hostOutput,inputLength * sizeof(DataType),cudaHostAllocDefault);
  cudaHostAlloc((void**)&resultRef,inputLength * sizeof(DataType),cudaHostAllocDefault);

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

  //@@ create streams 1-4
  cudaStream_t s[4];
  for(int i=0;i<4;++i)
  cudaStreamCreate(&s[i]);
  int S_seg=(inputLength-1)/N_seg+1;//divide input array into N subarrays, each has S_seg elements;
  
  double iStart = cpuSecond();
  for(int i=0;i<N_seg;++i){
    int si=i%4;//put this segment into si stream
    int off=i*S_seg;
    int bytes=min(inputLength-off,S_seg)*sizeof(DataType);
    cudaMemcpyAsync(&deviceInput1[off],&hostInput1[off],bytes,cudaMemcpyHostToDevice,s[si]);
    cudaMemcpyAsync(&deviceInput2[off],&hostInput2[off],bytes,cudaMemcpyHostToDevice,s[si]);
    vecAdd<<<(S_seg-1)/TPB+1,TPB,0,s[si]>>>(deviceInput1,deviceInput2,deviceOutput,off+bytes/sizeof(DataType),off);
    cudaMemcpyAsync(&hostOutput[off],&deviceOutput[off],bytes,cudaMemcpyDeviceToHost,s[si]);

  }
  cudaDeviceSynchronize();
  double iElaps = cpuSecond() - iStart;
  printf("%lf\n",iElaps);
  //for(int i=0;i<128;++i)
    //printf("%lf",hostOutput[i]);

  //@@ Insert code below to compare the output with the reference
  long double tmp=0;
  for(int i=0;i<inputLength;++i)
  {
  	tmp+=abs(resultRef[i]-hostOutput[i]);
  }
  tmp/=inputLength;

  printf("average difference:%LF\n",tmp);
  //@@ Free the GPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  cudaFreeHost(resultRef);
  //@@ Free the CPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  for(int i=0;i<4;++i)
    cudaStreamDestroy(s[i]);
  return 0;
}

