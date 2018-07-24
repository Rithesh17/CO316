// A0
// Q2. Write a CUDA program to calculate the sum of the elements in an array.

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

void arraySum (float * h_A, float h_sum, int n)
{
	int size = n * sizeof(float);
	float * d_A;
	float * d_sum;

	// 1. Allocate device memory (with error checking)
	cudaError_t err1 = cudaMalloc((void **) &d_A, size);
	if (err != cudaSuccess)
	{
		printf("%s in %s at line d.\n", cudaGetErrorString(err), _FILE_, _LINE_);
		exit(EXIT_FAILURE);
	}
	cudaError_t err2 = cudaMalloc((void *) &d_sum, sizeof(float));
	if (err != cudaSuccess)
	{
		printf("%s in %s at line d.\n", cudaGetErrorString(err), _FILE_, _LINE_);
		exit(EXIT_FAILURE);
	}

	// 2. Copy host memory to device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHosttoDevice);
	cudaMemcpy(d_sum, h_sum, sizeof(float), cudaMemcpyHosttoDevice);

	// 3. Initialize thread block and kernel grid dimensions
	dim3 DimGrid((n / 256;
	dim3 DimBlock(256);

	// 4. Invoke CUDA kernel
	arraySumKernel <<< DimGrid, DimBlock >>> (d_A, d_sum, n);

	// 5. Copy results from device to host
	float result;
	cudaMemcpy(&result, d_A, sizeof(float), cudaMemcpyDeviceToHost);

	// 6. Free device memory
	cudaFree(d);

	printf("Sum: %d\n", result);
}

// 7. CUDA kernel that computes the sum
__global__ 
void arraySumKernel (float * A, float * S, int N)
{
	__shared__ float sdata[256];

    	// each thread loads one element from global to shared mem
    	// note use of 1D thread indices (only) in this kernel
    	int i = blockIdx.x * blockDim.x + threadIdx.x;

    	sdata[threadIdx.x] = A[i];

    	__syncthreads();
    
	// do reduction in shared mem
    	for (int s = 1; s < blockDim.x; s *= 2)
    	{
        	int index = 2 * s * threadIdx.x;;

        	if (index < blockDim.x)
         	{
            		sdata[index] += sdata[index + s];
        	}
        	__syncthreads();
    	}

    	// write result for this block to global mem
    	if (threadIdx.x == 0)
        	atomicAdd(S,sdata[0]);	
}

// Main
int main()
{
	int n;
	printf("Number of elements: ");
	scanf("%d", &n);
	
	float *arr = (float *)malloc(n * sizeof(float));
	if (arr == NULL)
	{
		// malloc error
		return -1;
	}

	printf("Enter %d elements: ", n);
	int i;
	for (i = 0; i < n; i++)
	{
		scanf("%f", &arr[i]);
	}

	float sum = 0;
	arraySum(arr, sum, n);
	
	return 0;
}

























