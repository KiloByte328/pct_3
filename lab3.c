#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, commsize;
    int n = 28000;
    int m = 28000;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double C[n];
    double C_temp[n];
    double B[m];
    for (size_t k = 0; k < n; k++)
    {
        C_temp[k] = 0.0;
        B[k] = k;
    }
    double A[(n / commsize) * m];
    double start = MPI_Wtime();
    for (size_t j = 0; j < n / commsize; j++)
    {
        for (size_t i = 0; i < m; i++)
        {
            A[j * m + i] = i;
        }
        for (size_t i = 0; i < m; i++)
        {
            C_temp[rank * (n / commsize) + j] = C[j] + A[j * m + i] * B[i];
        }
    }
    MPI_Allreduce(&C_temp, &C, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double time = MPI_Wtime() - start;
    if (rank == 0)
        printf("DGEMV succesful, time is: %f\n", time);
    MPI_Finalize();
    return 0;
}