#include <wb.h>

#define NUM_BINS 4096
#define NUM_THREADS 512
#define WARP_SIZE 32
#define SM_SIZE 12288
#define BLOCKS_PER_SM 8
#define HIST_SIZE 128
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,\
   bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),\
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histAlgoMemPerBlock(unsigned int* devIn, \
  unsigned int* devOut, int inputLength, int R)
{
  __shared__ int hist_per_block[(NUM_BINS + 1) * R];

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int warp_block = blockidx.x / WARP_SIZE;

  int per_block_offset = (NUM_BINS + 1) * threadidx.x % R;

  int start = (inputLength / warp_block) * warp_id + WARP_SIZE * blockidx.x + lane;
  int finish = (inputLength / warp_block) * (warp_id + 1);
  int step = WARP_SIZE * gridDim.x;

  int i, sum, j;
  for(i = threadIdx.x; i < (NUM_BINS + 1) * R; i+= blockDim.x)
    hist_per_block[i] = 0;

  __syncthreads();

  for(i = start; i < finish; i += step)
    atomicAdd(&hist_per_block[per_block_offset + devIn[i]], 1);

  __syncthreads();

  for(i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
  {
    sum = 0;
    for(j = 0; j < (NUM_BINS + 1) * R; j += NUM_BINS + 1)
      sum += hist_per_block[i + j];
    atomicAdd(devOut + i, sum);
  }
}

int main(int argc, char *argv[])
{

    wbArg_t args;
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *deviceInput;
    unsigned int *deviceBins;
    int R;

    /* Read input arguments here */
    if(argc != 3)
    {
      printf("Usage: ./a.out <input.raw> <output.raw>\n");
      exit(1);
    }

    args = {argc, argv};

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), \
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

    R = SM_SIZE / (BLOCKS_PER_SM * (HIST_SIZE + 1));

    // Launch kernel
    // ----------------------------------------------------------
    wbLog(TRACE, "Launching kernel");
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Perform kernel computation here

    histAlgo <<<(int)ceil((float)inputLength / NUM_THREADS), NUM_THREADS>>> \
      (deviceInput, deviceBins, inputLength, R);

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
