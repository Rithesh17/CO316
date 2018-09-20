#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000
#define Nthreads 2

// Producer: fill an array of data
void fill_rand(int length, double *a)
{
	int i; 
   	for (i = 0; i < length; i++) 
	{
     		*(a + i) = (double) (rand() + rand()) / rand();
	}
}

// Consumer: sum the array
double Sum_array(int length, double *a)
{
	int i;  
	double sum = 0.0;
   	for (i = 0; i < length; i++)  
	{
		sum += *(a + i);
	}  
   	return sum; 
}
  
int main()
{
	printf("Producer - Consumer\n");
	double *A, sum, runtime;
  	int flag = 0, i;
	int numthreads;
	omp_set_num_threads(Nthreads);
	A = (double *) malloc(N * sizeof(double));

	// Start parallel execution
	#pragma omp parallel
  	{
		// Master block
     		#pragma omp master
     		{
        		numthreads = omp_get_num_threads();
			// There are two threads - producer and consumer
        		if(numthreads != 2)
        		{
           			printf("Error: Incorect number of threads, %d. \n",numthreads);
           			exit(-1);
        		}
        		runtime = omp_get_wtime();
     		}
     		#pragma omp barrier

     		#pragma omp sections
     		{
        		// Producer section
			#pragma omp section
        		{
           			fill_rand(N, A);
           			#pragma omp flush
           			flag = 1;
           			#pragma omp flush (flag)
        		}
			// Consumer section
        		#pragma omp section
        		{
           			#pragma omp flush (flag)
				// Wait (in an loop) for the producer to complete populating the array
           			while (flag != 1)
				{
              				#pragma omp flush (flag)
           			}
				#pragma omp flush
           			sum = Sum_array(N, A);
        		}
      		}
      		#pragma omp master
         	runtime = omp_get_wtime() - runtime;
   	}
	printf("Array produced:\n");
	for(i=0; i<N; i++)
		printf("%lf\n", A[i]);

	printf("\n\nSum of elements by the consumer: %lf\n", sum);
	printf("Runtime: %lf \n",runtime);
}
