// CUDA program to calculate the element-wise sum of two large arrays

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

    	if (i < n)
    	{
        	C[i] = A[i] + B[i];
    	}
}


// Main
int main(void)
{
    	// Error code to check return values for CUDA calls
    	cudaError_t err = cudaSuccess;

    	int num = 50000;
    	size_t size = num * sizeof(float);
    	printf("\n\tVector addition of %d elements\n\n", num);

    	// Allocate host memory (with error checking)
	printf("Allocating host memory...\n");
    	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
    	float *h_C = (float *)malloc(size);

    	if (h_A == NULL || h_B == NULL || h_C == NULL)
    	{
        	fprintf(stderr, "Failed to allocate host vectors!\n");
        	exit(EXIT_FAILURE);
    	}

    	// Initialize the host input vectors with random values
	printf("Initializing host input vectors...\n");
    	for (int i = 0; i < numElements; ++i)
    	{
        	h_A[i] = rand()/(float)RAND_MAX;
        	h_B[i] = rand()/(float)RAND_MAX;
    	}

    	// Allocate device memory (with error checking)
	printf("Allocating device memory...\n");
    	float *d_A = NULL;
    	err = cudaMalloc((void **)&d_A, size);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	float *d_B = NULL;
    	err = cudaMalloc((void **)&d_B, size);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	float *d_C = NULL;
    	err = cudaMalloc((void **)&d_C, size);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Copy the host input vectors A and B in host memory to the device input vectors in device memory
	printf("Copying input from host to device...\n");
    	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Launch CUDA Kernel
	printf("Launching vector addition kernel...\n");
    	int threadsPerBlock = 256;
    	int blocksPerGrid =(num + threadsPerBlock - 1) / threadsPerBlock;
    	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num);
   	err = cudaGetLastError();

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Copy device result vector in device memory to the host result vector in host memory.
	printf("Copying result from device to host...\n");
    	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Verify result
	printf("Verifying result...\n");
    	for (int i = 0; i < num; ++i)
    	{
        	if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        	{
            		fprintf(stderr, "Result verification failed at element %d!\n", i);
           		exit(EXIT_FAILURE);
        	}
    	}

	// Free device global memory
	printf("Freeing device memory...\n");
    	err = cudaFree(d_A);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	err = cudaFree(d_B);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	err = cudaFree(d_C);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Free host memory
	printf("Freeing host memory...\n");
    	free(h_A);
    	free(h_B);
    	free(h_C);

    	printf("Done.\n\n");
    	return 0;
}

