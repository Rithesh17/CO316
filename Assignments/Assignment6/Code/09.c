#include "mpi.h"
#include <stdio.h>

#define N 5

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank, size, i, j;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2)
  {
    printf("Please run with 2 processes.\n");
    MPI_Finalize();
    return 1;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Datatype type, type2;
  MPI_Type_contiguous(1, MPI_INT, &type2);
  MPI_Type_commit(&type2);

  int blocklen[N], displacement[N];
  for(i=0; i<N; i++)
  {
    blocklen[i] = N-i;
    displacement[i] = i*(N+1);
  }

  MPI_Type_indexed(N, blocklen, displacement, type2, &type);
  MPI_Type_commit(&type);

  int buffer[N*N];
  MPI_Status status;

  if (rank == 0)
  {
    printf("Original Data:\n");
    for (i=0; i<N; i++)
    {
      for(j=0; j<N; j++)
      {
        buffer[i*N+j] = i*N+j;
        printf("%4d", buffer[i*N+j]);
      }
      printf("\n");
    }

    MPI_Send(buffer, 1, type, 1, 123, MPI_COMM_WORLD);
  }

  if (rank == 1)
  {
    for (i=0; i<N*N; i++)
      buffer[i] = 0;

    MPI_Recv(buffer, 1, type, 0, 123, MPI_COMM_WORLD, &status);

    printf("\nAfter indexing:\n");

    for (i=0; i<N; i++)
    {
      for(j=0; j<N; j++)
        printf("%4d", buffer[i*N+j]);
      printf("\n");
      }
    fflush(stdout);
  }

  MPI_Finalize();
  return 0;
}
