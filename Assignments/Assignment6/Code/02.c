// 2. DAXPY Loop

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Vector size
#define len 65336

double X[len], Y[len], Xd[len], Yd[len];
int A = 2;

int main(int argc, char* argv[])
{
    	int rank, n;

	// Fill array
	for (int i = 0; i < len; i++)
    	{
         	X[i] = rand() % 5;
        	Xd[i] = X[i];
        	Y[i] = rand() % 5;
        	Yd[i] = Y[i];
    	}
	
    	MPI_Init(&argc, &argv);
    	MPI_Comm_size(MPI_COMM_WORLD, &n);
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) 
	{
		printf("Number of processes: %d\n", n);
	}
    
	double share = (double) len / n;
    
	// Initiate barrier synchronization
	MPI_Barrier(MPI_COMM_WORLD);

    	// Determine wall clock time
	double t1 = MPI_Wtime();
    
	for (int i = rank * share; i < (rank + 1) * share; i++)
    	{
        	X[i] = A * X[i] + Y[i];
    	}
    
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime() - t1;
    
	// printf("Rank: %d out of %d processes\n", rank, n);

    	MPI_Barrier(MPI_COMM_WORLD);
    	double t2 = MPI_Wtime();

    	if (rank == 0)
    	{
         	for (int i = 0; i < len; i++)
        	{
            		Xd[i] = A * Xd[i] + Yd[i];
        	}
    	}

    	MPI_Barrier(MPI_COMM_WORLD);
    	t2 = MPI_Wtime() - t2;

    	if (rank == 0)
    	{
        	printf("Time taken for MPI implementation: %lf\n", t1);
        	printf("Time taken for uniprocessor implementation: %lf\n", t2);
        	printf("Speedup: %lf\n", t2/t1);
    	}    

	MPI_Finalize();

    return 0;
}
