#include <wb.h>

#define NUM_BINS 4096
#define NUM_THREADS 512
#define LOG_WARP_SIZE 5

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, \
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

//The histogram algorithm requires some synchronisation primitives, which
//are not present in NVIDIA CUDA. To overcome the lack of synchronisation
//mechanisms, we have implemented a simulation of mutex through bounded
//waiting.
__global__ void histAlgoMutex(unsigned int* devIn, unsigned int* devOut, int inputLength)
{
  int tid = blockIdx.x * blockIdx.dim + threadIdx.x;
  unsigned int tag;
  volatile unsigned int bin = devIn[tid];

  do
  {
    //Since we are sure that the frequency of each bin does not exceed
    // 2 ^ (32 - LOG_WARP_SIZE)
    unsigned int val = devOut[bin] & (0xFFFFFFFF >> (32 - LOG_WARP_SIZE));

    tag = (threadIdx.x << (32 - LOG_WARP_SIZE)) | (val + 1);

    devOut[bin] = tag;

  } while(devOut[bin] != tag);
}

int main(int argc, char *argv[])
{
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  /* Read input arguments here */
  if(argc != 3)
  {
    printf("Usage: ./a.out <input.raw> <output.raw>\n");
    exit(1);
  }

  args = {argc, argv};

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),\
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(deviceInput, sizeof(unsigned int) * inputLength);
  cudaMalloc(deviceBins, sizeof(unsigned int) * NUM_BINS);

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemCpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, \
              cudaMemcpyHostToDevice);
  cudaMemCpy(deviceBins, 0, sizeof(unsigned int) * NUM_BINS, \
              cudaMemcpyHostToDevice);

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here

  histAlgo <<<(int)ceil((float)inputLength / NUM_THREADS), NUM_THREADS>>> \
    (deviceInput, deviceBins, inputLength);

  wbTime_stop(Compute, "Performing CUDA computation");
  // ----------------------------------------------------------

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemCpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, \
              cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Verify correctness
  // -----------------------------------------------------
  wbSolution(args, hostBins, NUM_BINS);

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceBins);
  cudaFree(deviceInput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  free(hostBins);
  free(hostInput);
  return 0;
}
