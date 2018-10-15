// 6. Collective Communication - Scatter - Gather

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int n, rank, i;
	float *sendbuf, *newarr, recvbuf;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n);

	if (rank == 0)
	{
		sendbuf = (float *) malloc(sizeof(float) * n);
		newarr = (float*) malloc(sizeof(float) * n);
		printf("Array: \n");
		for (i = 0; i < n; i++)
		{
			sendbuf[i] = rand() % 100;
			printf("%f\n", sendbuf[i]);
		}

	}
	
	// Entry i specifies the displacement (relative to sendbuf) from which to take the outgoing data to process i
	int *displs = (int *) malloc(sizeof(int) * n);

	// Size of each segment - number of elements to be sent to process P[i]
	int *sendcounts = (int *) malloc(sizeof(int) * n );

	for (i = 0; i < n; i++)
	{
		sendcounts[i] = 1;
		displs[i] = i;
	}

	// Scatter the array
	MPI_Scatterv(sendbuf, sendcounts, displs, MPI_FLOAT, &recvbuf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Each process finds the square root of each element it receives
	float root = sqrt(recvbuf);

	// Gathers data from all processes to the root process
	MPI_Gatherv(&root, 1, MPI_FLOAT, newarr, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("\nSquare root computation:\n");
		for(i = 0; i < n; i++)
		{
			printf("%f\n", newarr[i]);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();

	return 0;
}
