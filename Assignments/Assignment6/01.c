#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv)
{
  int rank, n, len;
  char name[100];

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Get_processor_name(name, &len);

  printf("Hello World! I am ranked %d among %d processes in %s processor\n", rank, n, name);

  MPI_Finalize();

  return 0;
}
