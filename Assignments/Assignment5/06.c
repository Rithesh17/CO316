#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

void main()
{
	printf("Calculation of pi - Worksharing and Reduction\n");
	double pi = 0.0, sum = 0.0;
	int i;
	step = 1.0 / (double)num_steps;
	double runtime = omp_get_wtime();
	#pragma omp parallel
	{
		double x;
		#pragma omp for reduction(+:sum)
		for (i = 0; i < num_steps; i++)
		{
			x = (i + 0.5) * step;
	 		sum += 4.0 / (1.0 + x * x);
		}
	}
	runtime = omp_get_wtime() - runtime;
	pi = step * sum;
	printf("Pi value: %f\tRuntime: %lf\n", pi, runtime); 
}


