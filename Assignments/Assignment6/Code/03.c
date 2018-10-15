#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[])
{

  int i;

  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  char* message;
  if (world_rank != 0)
  {
    message = "Hello World!";
    MPI_Send(message, strlen(message), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  else
  {
    for(i=1; i<world_size; i++)
    {
      MPI_Recv(message, 40, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Process 0 received message '%s' from process %d\n", message, i);
    }
  }

  MPI_Finalize();
  return 0;
}
