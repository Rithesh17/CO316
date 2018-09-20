#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define max_threads 1024
#define dim 1000

void main()
{
	printf("Matrix Multiply\n");
	int i, j, k, temp;
	int *a, *b, *c;
	a = (int *) malloc(dim * dim * sizeof(int));
      	b = (int *) malloc(dim * dim * sizeof(int));
	c = (int *) malloc(dim * dim * sizeof(int));	
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			*(a + (i * dim + j)) = rand();
 		}
	}
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			*(b + (i * dim + j)) = rand();
 		}
	}
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			*(c + (i * dim + j)) = 0;
 		}
	}

	double rununi = omp_get_wtime();
	for (i = 0; i < dim; i++)
	{
  		for (j = 0; j < dim; j++)
		{
    	 		temp = 0;
    			for (k = 0; k < dim; k++)
    			{
				temp += *(a + (i * dim + k)) * *(b + (k * dim + j));
    			}
			*(c + (i * dim + j)) = temp;
  		}	
	}
	rununi = omp_get_wtime() - rununi;
	printf("Thread: 1\tRuntime: %lf\n\n", rununi);
	
	for (i = 2; i <= max_threads; i *= 2)
	{
		omp_set_num_threads(i);
		double runtime = omp_get_wtime();
		#pragma omp parallel for private(i, j, k, temp) shared(a, b, c)
		for (i = 0; i < dim; i++)
		{
  			for (j = 0; j < dim; j++)
			{
    		 		temp = 0;
    				for (k = 0; k < dim; k++)
    				{
					temp += *(a + (i * dim + k)) * *(b + (k * dim + j));
    				}
				*(c + (i * dim + j)) = temp;
  			}	
		}	
		runtime = omp_get_wtime() - runtime;
		runtime = runtime / rununi;
		printf("Threads: %d\tSpeedup: %lf\n", i, runtime);
	}
}

