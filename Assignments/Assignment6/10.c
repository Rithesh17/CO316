// 10. Matrix Multiplication on a Cartesian Grid (2D Mesh) using Cannon’s Algorithm

// To run: mpirun -n 16 ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ndims 2			// The grid contains ndims dimensions
#define MATRIX_SIZE 8		// Multiple of 4

typedef struct
{
	int N; 			// The number of processors in a row (column).
	int size; 		// Number of processors. (Size = N * N)
	int row; 		// This processor’s row number.
	int col; 		// This processor’s column number.
	int MyRank; 		// This processor’s unique identifier.
	MPI_Comm Comm; 		// Communicator for all processors in the grid.
	MPI_Comm row_comm; 	// All processors in this processor’s row . 
	MPI_Comm col_comm; 	// All processors in this processor’s column. 
} grid_info;

void SetUp_Mesh(grid_info *grid)
{
	// Number of processes per dimension
	int dims[] = {4, 4};

	// Logical array of size ndims specifying whether the grid is periodic (true) or not (false) in each dimension
	int periods[] = {1, 1};

	// Records the position of the process in the grid
	int Coordinates[2];

	MPI_Comm_size(MPI_COMM_WORLD, &(grid -> size));
	MPI_Comm_rank(MPI_COMM_WORLD, &(grid -> MyRank));

	grid -> N = 4;

	// Create a new communicator to which a Cartesian 4 × 4 grid topology is attached
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &(grid -> Comm));

	// Determines process coords in cartesian topology given rank in group
	MPI_Cart_coords(grid -> Comm, grid -> MyRank, ndims, Coordinates);

	grid -> row = Coordinates[0];
	grid -> col = Coordinates[1];

	// Partition communicator into subgroups which form lower-dimensional cartesian subgrids

	// The i-th entry of remain_dims specifies whether the i-th dimension is kept in the subgrid (true) or is dropped (false)
	int remain_dims[2] = {0, 1};

	MPI_Cart_sub(grid -> Comm, remain_dims, &(grid -> row_comm));

	remain_dims[0] = 1;
	remain_dims[1] = 0;

	MPI_Cart_sub(grid -> Comm, remain_dims, &(grid -> col_comm));
}

int main (int argc, char *argv[])
{

	int i, j, k, l, x, y, index, istage, Proc_Id, Root = 0, block_size, matrix_block, lindex, gr, gc;
	int src, dest, send_tag, recv_tag, Bcast_root;

	int A[MATRIX_SIZE][MATRIX_SIZE], B[MATRIX_SIZE][MATRIX_SIZE], C[MATRIX_SIZE][MATRIX_SIZE];
	int *A_block, *B_block, *C_block, *Temp_BufferA;

	int *A_array, *B_array, *C_array;

	grid_info grid;
	MPI_Status status;

	MPI_Init (&argc, &argv);

	// Creation of a Grid of processes
	SetUp_Mesh(&grid);

	// In the root process, populate the multiplicand and multiplier arrays (A and B) with random numbers
	if (grid.MyRank == Root)
	{
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
			{
				A[i][j] = rand() % 5;
				B[i][j] = rand() % 5;
			}
		}

	}

	MPI_Barrier(grid.Comm);

	// Divide the array into equal sized blocks

	block_size = MATRIX_SIZE / grid.N;
	matrix_block = block_size * block_size;

	// Memory allocating for Block Matrices
	A_block = (int *) malloc (matrix_block * sizeof(int));
	B_block = (int *) malloc (matrix_block * sizeof(int));

	// Memory for arrangmeent of the data in one dimensional arrays before MPI_SCATTER
	A_array =(int *) malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	B_array =(int *) malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);

	// Rearrange the input matrices in one dimensional arrays by appropriate order
	if (grid.MyRank == Root)
	{
		// Rearranging Matrix A
		for (x = 0; x < grid.N; x++)
		{
			for (y = 0; y < grid.N; y++)
			{
				Proc_Id = x * grid.N + y;
				for (i = 0; i < block_size; i++)
				{
					gr = x * block_size + i;
					for (j = 0; j < block_size; j++)
					{
						lindex  = (Proc_Id * matrix_block) + (i * block_size) + j;
						gc = y * block_size + j;
						A_array[lindex] = A[gr][gc];
					}
				}
			}
		}

		// Rearranging Matrix B
		for (x = 0; x < grid.N; x++)
		{
			for (y = 0; y < grid.N; y++)
			{
				Proc_Id = x * grid.N + y;
				for (i = 0; i < block_size; i++)
				{
					gr = x * block_size + i;
					for (j = 0; j < block_size; j++)
					{
						lindex = (Proc_Id * matrix_block) + (i * block_size) + j;
						gc = y * block_size + j;
						B_array[lindex] = B[gr][gc];
					}
				}
			}
		}

	}

	MPI_Barrier(grid.Comm);

	// Scatter to each process in the grid
	MPI_Scatter (A_array, matrix_block, MPI_FLOAT, A_block, matrix_block, MPI_FLOAT, 0, grid.Comm);
	MPI_Scatter (B_array, matrix_block, MPI_FLOAT, B_block, matrix_block, MPI_FLOAT, 0, grid.Comm);


	// Initial arrangement of Matrices

	if (grid.row != 0)
	{
		src = (grid.col + grid.row) % grid.N;
		dest = (grid.col + grid.N - grid.row) % grid.N;
		recv_tag = 0;
		send_tag = 0;
		MPI_Sendrecv_replace(A_block, matrix_block, MPI_FLOAT, dest, send_tag, src, recv_tag, grid.row_comm, &status);
	}
	if (grid.col != 0)
	{
		src   = (grid.row + grid.col) % grid.N;
		dest = (grid.row + grid.N - grid.col) % grid.N;
		recv_tag = 0;
		send_tag = 0;
		MPI_Sendrecv_replace(B_block, matrix_block, MPI_FLOAT, dest,send_tag, src, recv_tag, grid.col_comm, &status);
	}

	// Allocate Memory for Block C Array
	C_block = (int *) malloc (block_size * block_size * sizeof(int));
	for (index = 0; index < block_size * block_size; index++)
	{
		C_block[index] = 0;
	}

	// Main loop

	send_tag = 0;
	recv_tag = 0;

	for (istage = 0; istage < grid.N; istage++)
	{
		index = 0;
		for (i = 0; i < block_size; i++)
		{
			for (j = 0; j < block_size; j++)
			{
				for (l = 0; l < block_size; l++)
				{
					C_block[index] += A_block[i * block_size + l] * B_block[l * block_size + j];
				}
				index++;
			}
		}
		// Move Block of Matrix A by one position left with wraparound
		src   = (grid.col + 1) % grid.N;
		dest = (grid.col + grid.N - 1) % grid.N;
		MPI_Sendrecv_replace(A_block, matrix_block, MPI_FLOAT, dest,send_tag, src, recv_tag, grid.row_comm, &status);

		// Move Block of Matrix B by one position upwards with wraparound
		src   = (grid.row + 1) % grid.N;
		dest = (grid.row + grid.N - 1) % grid.N;
		MPI_Sendrecv_replace(B_block, matrix_block, MPI_FLOAT, dest, send_tag, src, recv_tag, grid.col_comm, &status);
	}


	// Memory for output global matrix in the form of array
	if(grid.MyRank == Root)
	{
		C_array = (int *) malloc (sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	}

	MPI_Barrier(grid.Comm);

	// Gather output block matrices at processor 0
	MPI_Gather (C_block, block_size * block_size, MPI_FLOAT, C_array,block_size*block_size, MPI_FLOAT, Root, grid.Comm);


	// Rearranging the output matrix in a array by approriate order
	if (grid.MyRank == Root)
	{
		for (x = 0; x < grid.N; x++)
		{
			for (y = 0; y < grid.N; y++)
			{
				Proc_Id = x * grid.N + y;
				for (i = 0; i < block_size; i++)
				{
					gr = x * block_size + i;
					for (j = 0; j < block_size; j++)
					{
						lindex = (Proc_Id * block_size * block_size) + (i * block_size) + j;
						gc = y * block_size + j;
						C[gr][gc] = C_array[lindex];
					}
				}
			}
		}
		printf("Matrix A :\n");
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
				printf ("%d ", A[i][j]);
			printf ("\n");
		}
		printf("\n");

		printf("Matrix B : \n");
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
				printf("%d ", B[i][j]);
			printf("\n");
		}
		printf("\n");

		printf("Matrix C :\n");
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
				printf("%d ",C[i][j]);
			printf("\n");
		}
	}

	MPI_Finalize();
	return 0;
}
