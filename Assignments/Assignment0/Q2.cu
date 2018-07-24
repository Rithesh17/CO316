#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime_api.h>

__global__ void vecSum(double* devIn, int pow_step, int n)
{
	//The thread ID (including its block ID)
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//Safety check to prevent unwanted threads.
	if(pow_step*i < n)
		//The two 'adjacent' elements of the array (or 
		//the two children in the segment tree) are added and
		//the result is stored in the first element.
		devIn[pow_step*i] = devIn[pow_step*i+(pow_step/2)] + devIn[pow_step*i];
}

int main()
{
	//Size of the array
	int n = 15;
	
	//hostIn: The array accessible by the host.
	//devIn: The input array accessible by the device.
	double *hostIn, *devIn;
	//hostOut: The output value accessible by the host.
	double hostOut;

	//The total size of the array (in bytes)
	size_t b = n*sizeof(double);

	//Allocating memory to host and device copies of array
	hostIn = (double*)malloc(b);
	cudaMalloc(&devIn, b);

	//Initialising the array. Here, we are randomly initialising the values.
	int i;
	printf("\nArray: ");
	for(i=0; i<n; i++)
	{
		hostIn[i] = rand()%10 + (float)rand()/RAND_MAX;
		printf("%f ", hostIn[i]);
	}

	//Copying the values in the host array to the device memory.
	cudaMemcpy(devIn, hostIn, b, cudaMemcpyHostToDevice);

	//Defining the block size and the grid size.
	int blk_size = 8, grd_size = (int)ceil((float)n/blk_size);

	//We are constructing a segment tree of the given array, where the internal
	//nodes store the sum of the subarray corresponding to the leaves in its
	//subtree. Each level in the tree can then be used to exhibit data-level parallelism.

	//The step variable indicates the total levels of the tree.
	int step = (int)ceil((float)(log(n)/log(2)));

	for(i=0; i<step; i++)
		//We will be calling the device function corresponding to each level of the 
		//tree to achieve parallelism
		vecSum<<<grd_size, blk_size>>>(devIn, pow(2, i+1), n);
	
	//Copying the value of the output (which is present as the first element in the devIn array)
	//to the host memory.
	cudaMemcpy(&hostOut, &devIn[0], sizeof(double), cudaMemcpyDeviceToHost);

	printf("\n\nFinal sum: %f\n", hostOut);

	//Freeing the host and the device memory.
	cudaFree(devIn);
	free(hostIn);

	return 0;
}
