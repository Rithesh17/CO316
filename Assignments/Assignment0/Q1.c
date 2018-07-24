#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>

void getDevProp(struct cudaDeviceProp devProp)
{
        int i;
        printf("Major revision number:         %d\n",  devProp.major);
        printf("Minor revision number:         %d\n",  devProp.minor);
        printf("Name:                          %s\n",  devProp.name);
        printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
        printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
        printf("Warp size:                     %d\n",  devProp.warpSize);
        printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
        printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        int blk[3], grd[3];
        printf("Maximum dimensions of block:   [");
        for (i = 0; i < 3; ++i)
                printf("%d, ", devProp.maxThreadsDim[i]);
        printf("\b\b]\nMaximum dimensions of grid:    [");
        for (i = 0; i < 3; ++i)
                printf("%d, ", devProp.maxGridSize[i]);
        printf("\b\b]\nClock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
        printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        return;
}

int main()
{
        struct cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        getDevProp(devProp);

        return 0;
}
