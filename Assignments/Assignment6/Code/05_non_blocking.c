#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int half = world_size, Pn = world_rank, sum = 0;

  int array[world_size];

  for(int i=0;i<world_size;++i)
    array[i] = i;

  if(world_rank == 0)
  {
    printf("Calculating sum of array using non-blocking calls.\n");
    printf("Array:\n");

    for(int i=0;i<world_size;++i)
    {
      sum += array[i];
      printf("%d ", array[i]);
    }
    printf("\n");

    printf("Expected sum: %d\n\n", sum);

    sum = 0;
  }

  MPI_Status *status;
  MPI_Request request;

  do
  {
    if(half%2 != 0)
    {
      MPI_Isend(&array[half], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);
    }

    if (half%2 != 0 && Pn == 0)
    {
      int partial_sum;
      MPI_Irecv(&partial_sum, 1, MPI_INT, half, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);

      array[0] += partial_sum;
    }

    half /= 2;

    if (Pn < half)
    {
      int partial_sum;
      MPI_Irecv(&partial_sum, 1, MPI_INT, half + Pn, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);

      array[Pn] += partial_sum;
    }
    else
    {
      MPI_Isend(&array[Pn], 1, MPI_INT, Pn - half, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);
    }

  }while(half > 1);

  if(Pn == 0)
    printf("Obtained sum after computation: %d\n", array[0]);

  MPI_Finalize();
}
