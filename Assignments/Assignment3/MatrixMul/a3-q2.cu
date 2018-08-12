#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Number of threads in each dimension of the block.
#define THREAD_NUM 16

// CUDA kernel
__global__ void matrixMul(int *A, int *B, int *C, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int num = n;

    	if (row < num && col < num)
    	{
        	long Cvalue = 0;
		for (int i = 0; i < num; i++)
		{
			Cvalue += A[row * num + i] * B[i * num + col];
		}
		C[row * num + col] = Cvalue;
    	}
}


// Main
int main(void)
{
    	// Error code to check return values for CUDA calls
    	cudaError_t err = cudaSuccess;

    	int num = 512, i, j;
    	size_t size = num * num * sizeof(int);
    	printf("\n\tMatrix multiplication of two %d * %d matrices\n\n", num, num);

    	int h_A[num][num], h_B[num][num], h_C[num][num];
	
	printf("Initializing host input vectors...\n");
    	for (int i = 0; i < num; i++)
    	{
		for (int j = 0; j < num; j++)
        	{
			//h_A[i][j] = rand();
        		//h_B[i][j] = rand();
			h_A[i][j] = 1;
			h_B[i][j] = 1;
		}
    	}

    	// Allocate device memory (with error checking)
	printf("Allocating device memory...\n");
    	int *d_A = NULL;
    	err = cudaMalloc((void **)&d_A, size);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	int *d_B = NULL;
    	err = cudaMalloc((void **)&d_B, size);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	int *d_C = NULL;
    	err = cudaMalloc((void **)&d_C, size);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Copy from host to device
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
	printf("Displaying the input matrices...\n\nMatrix A: \n");

        for(i=0; i<num; i++){
                for(j=0; j<num; j++)
                        printf("%d ", h_A[i][j]);
                printf("\n");
        }
        printf("\nMatrix B: ");

        for(i=0; i<num; i++){
                for(j=0; j<num; j++)
                        printf("%d ", h_B[i][j]);
                printf("\n");
        }
        printf("\n");
/*
        printf("\nd_b:\n");

        for(i=0; i<num; i++){
                for(j=0; j<num; j++)
                        printf("%d ", d_B[i*num+j]);
                printf("\n");
        }
        printf("\n");
*/
    	// Launch CUDA Kernel
	printf("Launching vector multiplication kernel...\n");
	dim3 dimBlock(THREAD_NUM, THREAD_NUM, 1);
    	dim3 dimGrid((int) ceil((float)num/dimBlock.x), (int) ceil((float)num/dimBlock.y), 1);
    	matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, num);
   	err = cudaGetLastError();

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Copy result from device to host
	printf("Copying result from device to host...\n");
    	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
/*
	printf("d_c:\n");

        for(i=0; i<num; i++){
                for(j=0; j<num; j++)
                        printf("%d ", d_C[i*num+j]);
                printf("\n");
        }
*/
        printf("Displaying the output matrix...\n\nMatrix C: \n");

	for(i=0; i<num; i++){
		for(j=0; j<num; j++)
			printf("%d ", h_C[i][j]);
		printf("\n");
	}
	printf("\n");

	// Free device global memory
	printf("Freeing device memory...\n");
    	err = cudaFree(d_A);
	/*if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}*/

    	err = cudaFree(d_B);
	/*if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}*/

    	err = cudaFree(d_C);
	/*if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}*/

    	printf("Done.\n\n");
    	return 0;
}
