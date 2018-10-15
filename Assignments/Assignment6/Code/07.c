#include "mpi.h"
#include <stdio.h>

struct Partstruct
{
    char c;
    int i[2];
    float f[4];
};

int main(int argc, char *argv[])
{
    struct Partstruct particle[2];

    int i, j, world_rank, world_size;
    MPI_Status status;
    MPI_Datatype Particletype;

    MPI_Datatype type[3] = { MPI_CHAR, MPI_DOUBLE, MPI_CHAR };
    int blocklen[3] = { 1, 2, 4 };
    MPI_Aint disp[3];

    MPI_Init(&argc, &argv);

    disp[0] = 0;
    disp[1] = 4;
    disp[2] = 20;

    MPI_Type_create_struct(3, blocklen, disp, type, &Particletype);
    MPI_Type_commit(&Particletype);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    particle[0].c = 'z';
    particle[0].i[0] = -1; particle[0].i[1] = -1;
    for(i=0; i<4; i++)
      particle[0].f[i] = 0;

    if(world_rank == 0)
    {
      particle[0].c = 'a';
      particle[0].i[0] = 0; particle[0].i[1] = 1;
      for(i=0; i<4; i++)
        particle[0].f[i] = i;

      particle[1].c = 'b';
      particle[1].i[0] = 10; particle[1].i[1] = 11;
      for(i=0; i<3; i++)
        particle[1].f[i] = 10+i;
      particle[1].f[3] = 0;
    }

    printf("Process %d: Data before broadcast: {'%c', [%d, %d], \
[%f, %f, %f, %f]}\n", world_rank, particle[0].c, particle[0].i[0], particle[0].i[1],\
particle[0].f[0], particle[0].f[1], particle[0].f[2], particle[0].f[3]);
    MPI_Bcast(&particle[0], 1, Particletype, 0, MPI_COMM_WORLD);

    printf("Process %d: Broadcast message recieved. Data: {'%c', [%d, %d], \
[%f, %f, %f, %f]}\n", world_rank, particle[0].c, particle[0].i[0], particle[0].i[1],\
particle[0].f[0], particle[0].f[1], particle[0].f[2], particle[0].f[3]);

    if (world_rank == 0)
    {
      for(i=1; i < world_size; i++)
        MPI_Send(&particle[1], 1, Particletype, i, 1, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(&particle[1], 1, Particletype, 0, 1, MPI_COMM_WORLD, &status);

        printf("Process %d: P2P message recieved. Data: {'%c', [%d, %d], \
[%f, %f, %f, %f]}\n", world_rank, particle[1].c, particle[1].i[0], particle[1].i[1],\
particle[1].f[0], particle[1].f[1], particle[1].f[2], particle[1].f[3]);

    }

    MPI_Finalize();
    return 0;
}
