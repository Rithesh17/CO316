#include <stdio.h>
#include <omp.h>

void printHello(int id)
{
	printf("Hello World from thread %d.\n", id);
}

void main()
{
	double runtime = omp_get_wtime();
	#pragma omp parallel
	{
		int threadID = omp_get_thread_num();
		printHello(threadID);
	}
	runtime = omp_get_wtime() - runtime;
	printf("Runtime: %lf\n", runtime);
}

